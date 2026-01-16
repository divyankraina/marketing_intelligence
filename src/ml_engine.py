import pandas as pd
import numpy as np
import joblib
import os
import shap
import torch
import xgboost as xgb
import lightgbm as lgb
from scipy.stats import ks_2samp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold, cross_validate, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import RobustScaler, QuantileTransformer, OneHotEncoder, PolynomialFeatures, FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectFromModel, SelectKBest
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.cluster import KMeans
from category_encoders import TargetEncoder, MEstimateEncoder
from sentence_transformers import SentenceTransformer
from src.monitoring import PerformanceMonitor

MODEL_PATH = "models/discount_predictor.joblib"
DRIFT_PATH = "models/drift_stats.joblib"
EXPLAINER_PATH = "models/surrogate_explainer.joblib"

# ===============
# 1. Transformers 
# ===============

class DataCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        def clean_currency(val):
            if isinstance(val, str):
                val = val.replace('₹', '').replace(',', '').strip()
                try: return float(val)
                except ValueError: return np.nan
            return val
        def clean_rating(val):
            if isinstance(val, str):
                if '|' in val: return np.nan
                try: return float(val.replace(',', ''))
                except ValueError: return np.nan
            return val

        if 'actual_price' in X.columns: X['actual_price'] = X['actual_price'].apply(clean_currency)
        if 'rating' in X.columns: X['rating'] = X['rating'].apply(clean_rating)
        if 'rating_count' in X.columns: X['rating_count'] = X['rating_count'].apply(clean_currency)
        return X

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        if 'category' in X.columns:
            cat_split = X['category'].astype(str).str.split('|', expand=True).iloc[:, :7]
            cat_split.columns = [f'cat_l{i}' for i in range(len(cat_split.columns))]
            for i in range(7):
                col_name = f'cat_l{i}'
                if col_name not in cat_split.columns: cat_split[col_name] = "Missing"
                else: cat_split[col_name] = cat_split[col_name].fillna("Missing").str.strip()
            X = pd.concat([X, cat_split], axis=1)
        else:
            for i in range(7): X[f'cat_l{i}'] = "Missing"

        if 'product_name' in X.columns:
            X['brand'] = X['product_name'].astype(str).apply(lambda x: x.split()[0].strip() if len(x.split()) > 0 else 'Generic')
            X['name_len'] = X['product_name'].astype(str).apply(len)
        else:
            X['brand'] = 'Generic'; X['name_len'] = 0
            
        if 'about_product' in X.columns:
            X['desc_len'] = X['about_product'].astype(str).apply(len)
        else:
            X['desc_len'] = 0

        X['price_per_rating'] = X['actual_price'] / (X['rating'] + 1e-5)
        X['popularity_score'] = np.log1p(X['rating_count'].fillna(0)) * X['rating'].fillna(0)
        X['combined_text'] = (X['brand'] + " " + X['cat_l1'] + " " + X.get('product_name', pd.Series(['']*len(X))).fillna(''))
        return X

class DenseEmbeddingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
    def fit(self, X, y=None): return self
    def transform(self, X):
        if self.model is None:
            self.model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        if isinstance(X, pd.DataFrame): text_data = X.iloc[:, 0].astype(str).tolist()
        else: text_data = X.astype(str).tolist()
        return self.model.encode(text_data, convert_to_numpy=True, show_progress_bar=True)

class KMeansFeaturizer(BaseEstimator, TransformerMixin):
    def __init__(self, k=10):
        self.k = k
        self.kmeans = None
        self.scaler = RobustScaler()
    def fit(self, X, y=None):
        X_scaled = self.scaler.fit_transform(X)
        self.kmeans = KMeans(n_clusters=self.k, random_state=42, n_init=10)
        self.kmeans.fit(X_scaled)
        return self
    def transform(self, X):
        X_scaled = self.scaler.transform(X)
        return self.kmeans.predict(X_scaled).reshape(-1, 1)

# ==========================================
# 2. Main Predictor 
# ==========================================

class DiscountPredictor:
    def __init__(self):
        self.model = None
        self.drift_detector = None 
        self.surrogate_pipeline = None  
        self.explainer = None
        self.fitted_selector = None
        
        self.best_params = {
            'xgb': {'n_estimators': 300, 'learning_rate': 0.05, 'max_depth': 6},
            'lgb': {'n_estimators': 300, 'learning_rate': 0.05, 'num_leaves': 31},
            'rf': {'n_estimators': 200, 'max_depth': 12}
        }

    def _get_preprocessor_pipeline(self):
        numeric_features = ['actual_price', 'rating_count', 'rating', 'price_per_rating', 'popularity_score']
        numeric_transformer = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', QuantileTransformer(output_distribution='normal', random_state=42))])

        cluster_features = ['actual_price', 'rating', 'rating_count']
        cluster_transformer = Pipeline([('imputer', SimpleImputer(strategy='median')), ('kmeans', KMeansFeaturizer(k=10)), ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

        categorical_features = [f'cat_l{i}' for i in range(7)] + ['brand']
        categorical_transformer = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='Missing')), ('target_enc', TargetEncoder())])

        tfidf_transformer = Pipeline([('tfidf', TfidfVectorizer(max_features=200))])
        bert_transformer = Pipeline([('embd', DenseEmbeddingTransformer())])
        poly_transformer = Pipeline([('poly', PolynomialFeatures(degree=2, interaction_only=False))])
        log_transformer = Pipeline([('log', FunctionTransformer(np.log1p))])
        sqrt_transformer = Pipeline([('sqrt', FunctionTransformer(np.sqrt))])
        return Pipeline([
            ('cleaner', DataCleaner()),
            ('engineer', FeatureEngineer()),
            ('preprocessor', ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features),
                    ('clusters', cluster_transformer, cluster_features),
                    ('cat', categorical_transformer, categorical_features),
                    ('poly',poly_transformer, numeric_features),
                    ('log', log_transformer, ['actual_price', 'rating']),
                    ('sqrt', sqrt_transformer, ['actual_price', 'rating']),
                    ('tfidf', tfidf_transformer, 'combined_text'),
                    ('embd', bert_transformer, 'combined_text')
                ]
            ))
        ])

    def _tune_base_models(self, X_processed, y):
        print("   ✂️  Running Feature Selection (Robust ElasticNetCV)...")
        
        
        selection_model = ElasticNetCV(
            cv=5, 
            random_state=42, 
            max_iter=10000,   
            tol=1e-3,         
            l1_ratio=0.9
        )
        
        selector = SelectFromModel(selection_model, threshold="median", max_features=100)
        X_selected = selector.fit_transform(X_processed, y)
        self.fitted_selector = selector
        
        n_dropped = X_processed.shape[1] - X_selected.shape[1]
        print(f"      📉 Dropped {n_dropped} noisy features. Tuning on {X_selected.shape[1]} features.")
        print(X_selected)
        print("   🔎 Tuning Hyperparameters (RandomizedSearch)...")
        
        # XGBoost
        xgb_grid = {'n_estimators': [200, 500, 800, 1500], 'learning_rate': [0.001,0.01, 0.05, 0.1], 'max_depth': [4, 6, 8, 12]}
        search = RandomizedSearchCV(xgb.XGBRegressor(n_jobs=-1, random_state=42), xgb_grid, n_iter=5, cv=5, scoring='neg_root_mean_squared_error', n_jobs=1, verbose=3)
        search.fit(X_selected, y)
        self.best_params['xgb'] = search.best_params_
        print(f"      ✅ XGB Best: {search.best_params_}")

        # LightGBM
        lgb_grid = {'n_estimators': [200, 500, 1000, 1500], 'learning_rate': [0.001,0.01, 0.05], 'num_leaves': [31, 50, 100, 200], 'force_row_wise': [True]}
        search = RandomizedSearchCV(lgb.LGBMRegressor(verbose=-1, random_state=42), lgb_grid, n_iter=5, cv=5, scoring='neg_root_mean_squared_error', n_jobs=1, verbose=3)
        search.fit(X_selected, y)
        self.best_params['lgb'] = search.best_params_
        print(f"      ✅ LGBM Best: {search.best_params_}")

        # RF
        rf_grid = {'n_estimators': [100, 200, 500, 1000, 2000], 'max_depth': [10, 20, 30, 40, 50]}
        search = RandomizedSearchCV(RandomForestRegressor(n_jobs=-1, random_state=42), rf_grid, n_iter=5, cv=5, scoring='neg_root_mean_squared_error', n_jobs=1, verbose=3)
        search.fit(X_selected, y)
        self.best_params['rf'] = search.best_params_

    def train(self, df, tune=True):
        print("🚀 Starting Production Training Pipeline...")
        from src.ml_engine import DriftDetector
        if self.drift_detector is None: self.drift_detector = DriftDetector()

        if 'discount_percentage' in df.columns:
            y = df['discount_percentage'].astype(str).str.replace('%', '').str.strip()
            y = pd.to_numeric(y, errors='coerce').fillna(0)
            X = df.drop(columns=['discount_percentage'])
            mask = ~np.isnan(y)
            X = X[mask]; y = y[mask]
        else:
            raise ValueError("Dataset missing 'discount_percentage'")

        print("⚙️  Preprocessing Data & Generating Embeddings...")
        preprocessor_pipe = self._get_preprocessor_pipeline()
        X_processed = preprocessor_pipe.fit_transform(X, y)

        if tune:
            self._tune_base_models(X_processed, y)
            selector_step = self.fitted_selector
        else:
            # Fallback selector if tuning skipped
            selector_step = SelectFromModel(ElasticNetCV(cv=5, random_state=42, max_iter=10000), threshold="median")

        # 4. Final Pipeline
        estimators = [
            ('xgb', xgb.XGBRegressor(**self.best_params['xgb'], n_jobs=-1, random_state=42)),
            ('lgb', lgb.LGBMRegressor(**self.best_params['lgb'], verbose=-1, random_state=42)),
            ('rf', RandomForestRegressor(**self.best_params['rf'], n_jobs=-1, random_state=42)),
            ('ridge', RidgeCV(alphas=[0.1, 1.0, 10.0]))
        ]

        final_pipeline = Pipeline([
            ('preprocessor_pipe', preprocessor_pipe),
            ('selector', selector_step),
            ('regressor', StackingRegressor(
                estimators=estimators, 
                final_estimator=ElasticNetCV(cv=5, max_iter=5000, l1_ratio=[.1, .5, .7, .9, .95, .99, 1]), 
                cv=5, n_jobs=1, verbose=3
            ))
        ])

        self.model = TransformedTargetRegressor(regressor=final_pipeline, func=np.log1p, inverse_func=np.expm1)
        
        print("📊 Calculating Metrics (CV on Full Model)...")
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        scoring = [
            'r2',
            'neg_mean_squared_error',
            'neg_mean_absolute_error',
            'neg_root_mean_squared_error',
            'neg_mean_absolute_percentage_error',
            'max_error']
        scores = cross_validate(self.model, X, y, cv=cv, scoring=scoring, n_jobs=1, verbose=3)
        
        for item in scores:
            print(f"   🔹 {item}: {abs(scores[item].mean()):.4f} (±{abs(scores[item].std()):.4f})")
        
        PerformanceMonitor.update_accuracy(abs(scores['test_r2'].mean()))
        
        print("⚙️  Fitting Final Model...")
        self.model.fit(X, y)
        self.drift_detector.fit(df)
        
        self._train_surrogate(X, y)
        
        os.makedirs("models", exist_ok=True)
        joblib.dump(self.model, MODEL_PATH)
        joblib.dump(self.drift_detector, DRIFT_PATH)
        print("✅ All Artifacts Saved.")

    def _train_surrogate(self, X, y):
        print("💡 Fitting Surrogate Explainer...")
        surrogate_prep = ColumnTransformer([
            ('num', SimpleImputer(strategy='median'), ['actual_price', 'rating', 'rating_count']),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['cat_l0', 'cat_l1'])
        ])
        
        self.surrogate_pipeline = Pipeline([
            ('cleaner', DataCleaner()),
            ('engineer', FeatureEngineer()),
            ('prep', surrogate_prep),
            ('xgb', xgb.XGBRegressor(n_estimators=100, max_depth=3, random_state=42))
        ])
        
        cols = [c for c in ['actual_price', 'rating', 'rating_count', 'category', 'product_name'] if c in X.columns]
        self.surrogate_pipeline.fit(X[cols], y)
        
        self.explainer = shap.TreeExplainer(self.surrogate_pipeline.named_steps['xgb'])
        joblib.dump({'pipeline': self.surrogate_pipeline, 'explainer': self.explainer}, EXPLAINER_PATH)

    def load(self):
        from src.ml_engine import DriftDetector
        if self.drift_detector is None: self.drift_detector = DriftDetector()
        if os.path.exists(MODEL_PATH): self.model = joblib.load(MODEL_PATH)
        if os.path.exists(DRIFT_PATH): self.drift_detector = joblib.load(DRIFT_PATH)
        if os.path.exists(EXPLAINER_PATH):
            data = joblib.load(EXPLAINER_PATH)
            self.surrogate_pipeline = data['pipeline']
            self.explainer = data['explainer']

    def _validate_and_sanitize_input(self, input_data: dict):
        """Safety validation: sanitize and validate input data"""
        sanitized = input_data.copy()
        
        # Ensure required fields exist
        expected_cols = ['category', 'actual_price', 'rating', 'rating_count', 'product_name', 'about_product', 'review_content']
        for col in expected_cols:
            if col not in sanitized:
                sanitized[col] = "" if "price" not in col and "rating" not in col else 0
        
        # Sanitize numeric fields
        if isinstance(sanitized.get('actual_price'), (int, float)):
            sanitized['actual_price'] = max(0, float(sanitized['actual_price']))
        else:
            sanitized['actual_price'] = 0
        
        if isinstance(sanitized.get('rating'), (int, float)):
            sanitized['rating'] = max(0, min(5, float(sanitized['rating'])))  # Clamp to 0-5
        else:
            sanitized['rating'] = 0
        
        if isinstance(sanitized.get('rating_count'), (int, float)):
            sanitized['rating_count'] = max(0, int(sanitized['rating_count']))
        else:
            sanitized['rating_count'] = 0
        
        # Sanitize text fields
        for col in ['product_name', 'category', 'about_product', 'review_content']:
            if not isinstance(sanitized.get(col), str):
                sanitized[col] = str(sanitized.get(col, ""))
            # Limit text length to prevent abuse
            sanitized[col] = sanitized[col][:1000]
        
        return sanitized

    def predict(self, input_data: dict):
        """
        Accepts a dictionary of inputs to match the training dataframe structure.
        Includes safety validation.
        """
        if not self.model: raise Exception("Model not loaded")
        
        # Safety validation
        sanitized = self._validate_and_sanitize_input(input_data)
        data = pd.DataFrame([sanitized])
        
        prediction = self.model.predict(data)[0]
        
        # Safety: clamp prediction to reasonable bounds
        prediction = max(0, min(100, float(prediction)))
        
        return prediction

    def explain(self, input_data: dict):
        """
        Updated explainer to use full text context
        """
        if not self.explainer: return {"error": "Explainer not initialized"}
        
        data = pd.DataFrame([input_data])
        
        try:
            d = self.surrogate_pipeline.named_steps['cleaner'].transform(data)
            d = self.surrogate_pipeline.named_steps['engineer'].transform(d)
            d = self.surrogate_pipeline.named_steps['prep'].transform(d)
            
            shap_vals = self.explainer.shap_values(d)
            
            cat_names = self.surrogate_pipeline.named_steps['prep'].named_transformers_['cat'].get_feature_names_out(['cat_l0', 'cat_l1'])
            names = ['actual_price', 'rating', 'rating_count'] + list(cat_names)
            
            impacts = {}
            if len(shap_vals[0]) == len(names):
                for i in np.argsort(np.abs(shap_vals[0]))[::-1][:5]: 
                    name_clean = names[i].replace("cat_l0_", "Category: ").replace("cat_l1_", "Sub-Cat: ")
                    impacts[name_clean] = f"{shap_vals[0][i]:.2f} impact"
            
            return {"base": f"{self.explainer.expected_value:.2f}", "drivers": impacts}
        except Exception as e:
            return {"error": f"Explanation failed: {str(e)}"}

    def check_and_retrain(self, new_df_batch):
        if self.drift_detector.check_drift(new_df_batch):
            self.train(new_df_batch)
            return True
        return False
    
    def evaluate(self, test_df):
        """Evaluate model performance on test data"""
        if 'discount_percentage' not in test_df.columns:
            raise ValueError("Test data must contain 'discount_percentage' column")
        
        y_true = test_df['discount_percentage'].astype(str).str.replace('%', '').str.strip()
        y_true = pd.to_numeric(y_true, errors='coerce').fillna(0)
        X_test = test_df.drop(columns=['discount_percentage'])
        
        mask = ~np.isnan(y_true)
        X_test = X_test[mask]
        y_true = y_true[mask]
        
        if len(X_test) == 0:
            raise ValueError("No valid test samples after cleaning")
        
        # Predict using the model (which expects DataFrame)
        y_pred = self.model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        return {
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'n_samples': int(len(y_true))
        }
# ==========================================
# 2. Drift Detector
# ==========================================

class DriftDetector:
    def __init__(self):
        self.reference_stats = {}
    
    def fit(self, df):
        clean_df = DataCleaner().transform(df)
        self.reference_stats = {
            'actual_price': clean_df['actual_price'].dropna().tolist(),
            'rating': clean_df['rating'].dropna().tolist(),
        }
    
    def check_drift(self, new_df, p_value_threshold=0.01):
        if not self.reference_stats: return False 
        clean_new = DataCleaner().transform(new_df)
        drift_detected = False
        
        for col in ['actual_price', 'rating']:
            ref_data = self.reference_stats.get(col)
            new_data = clean_new[col].dropna().tolist()
            if ref_data and new_data:
                stat, p_val = ks_2samp(ref_data, new_data)
                if p_val < p_value_threshold:
                    drift_detected = True
                    PerformanceMonitor.log_drift()
        return drift_detected

