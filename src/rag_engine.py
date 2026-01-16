import os
import torch
import pandas as pd
import numpy as np
import re
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from peft import PeftModel
from src.config import Config

class DataIngestionUtils:
    """Helper class to ensure data is strictly typed before storage."""
    
    @staticmethod
    def clean_currency(val):
        """Robustly converts '₹1,299.00' or '1,299' to float 1299.0"""
        if isinstance(val, (int, float)): return float(val)
        if pd.isna(val) or val == "": return 0.0
        # Remove non-numeric chars except dot
        clean = re.sub(r'[^\d.]', '', str(val))
        try:
            return float(clean)
        except ValueError:
            return 0.0

    @staticmethod
    def clean_rating(val):
        """Converts '4.5|5' or '4.3' to float 4.3"""
        if isinstance(val, (int, float)): return float(val)
        val = str(val).split('|')[0] 
        return DataIngestionUtils.clean_currency(val)

class MarketingAssistant:
    def __init__(self):
        self.model_id = Config.LLM_MODEL_ID
        self.pipe = None
        self.df = None  

    def _prepare_grounded_text(self, row):
        """
        Serializes ALL columns into a dense text block.
        Example: "Product Name: iPhone 13 | Price: 59000 | Rating: 4.8..."
        """
        text_parts = []
        for col, val in row.items():
            if col in ['combined_text'] or pd.isna(val) or str(val).strip() == "":
                continue
            
            clean_key = str(col).replace('_', ' ').title()
            text_parts.append(f"{clean_key}: {str(val).strip()}")
        
        return "\n".join(text_parts)

    def ingest_data(self, df):
        print(f"📚 Ingesting with SOTA Embeddings ({Config.EMBEDDING_MODEL})...")
        
        # 1. Data Cleaning
        if 'discounted_price' in df.columns:
            df['discounted_price'] = df['discounted_price'].apply(DataIngestionUtils.clean_currency)
        if 'actual_price' in df.columns:
            df['actual_price'] = df['actual_price'].apply(DataIngestionUtils.clean_currency)
        if 'rating' in df.columns:
            df['rating'] = df['rating'].apply(DataIngestionUtils.clean_rating)
        if 'rating_count' in df.columns:
            df['rating_count'] = df['rating_count'].apply(DataIngestionUtils.clean_currency)
            
        # 2. Dedup & Save Structured Store
        if 'product_id' in df.columns:
            df = df.drop_duplicates(subset=['product_id'])
        
        # Save to Parquet
        print(f"   💾 Saving Structured Metadata to {Config.METADATA_STORE_PATH}...")
        df.to_parquet(Config.METADATA_STORE_PATH)
        self.df = df
        
        # 3. Vectorize for Semantic Search
        print("   ⚡ Generating Vectors...")
        records = df.to_dict(orient='records')
        docs = []
        for row in records:
            content = self._prepare_grounded_text(row)
            # We store product_id to link back to the DataFrame if needed
            meta = {"source": str(row.get('product_id', 'unknown'))}
            docs.append(Document(page_content=content, metadata=meta))
        
        print("   ⚡ Generating Dense Vectors...")
        embeddings = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL,
            model_kwargs={'device': 'cuda'},
            encode_kwargs={'normalize_embeddings': True} 
        )
        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local(Config.INDEX_PATH)
        print(f"✅ Ingestion Complete. Indexed {len(docs)} items.")

    def _get_deterministic_results(self, query):
        """
        Rule-Based Router. Returns DataFrame rows if a sort/filter is detected.
        Returns None if semantic search is required.
        """
        if self.df is None: return None
        q = query.lower()
        
        # --- LOGIC MAPPING ---
        # Map user phrases to (Column, AscendingBoolean)
        sort_rules = {
            "highest price": ("discounted_price", False),
            "most expensive": ("discounted_price", False),
            "lowest price": ("discounted_price", True),
            "cheapest": ("discounted_price", True),
            "highest rating": ("rating", False),
            "best rated": ("rating", False),
            "highest rated": ("rating", False),
            "worst rated": ("rating", True),
            "most reviews": ("rating_count", False),
            "highest review count": ("rating_count", False),
            "highest rating count": ("rating_count", False),
            "most popular": ("rating_count", False)
        }

        for phrase, (col, asc) in sort_rules.items():
            if phrase in q and col in self.df.columns:
                print(f"   🎯 Deterministic Router: Sorting by '{col}' (Asc: {asc})")
                # Filter out zeros/NaNs for valid sorting
                valid_df = self.df[self.df[col] > 0]
                return valid_df.sort_values(by=col, ascending=asc).head(5)

        return None

    def load_pipeline(self):
        print("Loading LLM pipeline...")
        # Load Structured Data
        if self.df is None and os.path.exists(Config.METADATA_STORE_PATH):
            self.df = pd.read_parquet(Config.METADATA_STORE_PATH)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, 
            bnb_4bit_use_double_quant=True, 
            bnb_4bit_quant_type="nf4", 
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id, 
            quantization_config=bnb_config, 
            device_map="auto",
            trust_remote_code=True
        )
        
        if os.path.exists(Config.ADAPTER_PATH):
            print(f"✅ Found Adapter. Loading...")
            model = PeftModel.from_pretrained(model, Config.ADAPTER_PATH)

        self.pipe = pipeline(
            "text-generation", 
            model=model, 
            tokenizer=tokenizer, 
            max_new_tokens=512, 
            temperature=0.1,
            repetition_penalty=1.1
        )

    def answer(self, query):
        if not hasattr(self, 'pipe'): self.load_pipeline()
        
        # 1. Try Deterministic (Structured) Search First
        top_rows = self._get_deterministic_results(query)
        
        # 2. Fallback to Vector (Semantic) Search
        if top_rows is None:
            print("   🧠 Semantic Router: Using Vector Search")
            if not os.path.exists(Config.INDEX_PATH): raise Exception("Index missing.")
            
            embeddings = HuggingFaceEmbeddings(
                model_name=Config.EMBEDDING_MODEL, 
                model_kwargs={'device': 'cuda'}
            )
            # Re-load index (safely)
            vectorstore = FAISS.load_local(Config.INDEX_PATH, embeddings)
            docs = vectorstore.similarity_search(query, k=4)
            context_list = [d.page_content for d in docs]
        else:
            # Convert DataFrame rows to Text Context
            context_list = [self._prepare_grounded_text(row) for _, row in top_rows.iterrows()]

        # 3. Generate Answer
        context_text = "\n---\n".join(context_list)
        
        prompt = (
            f"<s>[INST] You are a helpful assistant. Answer the question using ONLY the context provided below.\n"
            f"Do not make up facts. If the answer is not in the context, say 'I don't know'.\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {query} [/INST]"
        )
        
        result = self.pipe(prompt)
        final_answer = result[0]['generated_text'].split("[/INST]")[-1].strip()
        
        return {"answer": final_answer, "context": context_list}
    
    def evaluate_grounding_accuracy(self, test_queries: list):
        """
        Evaluate RAG grounding accuracy by checking if answers are factually correct
        based on the retrieved context.
        
        Args:
            test_queries: List of dicts with 'query' and 'expected_answer' keys
            
        Returns:
            Dictionary with accuracy metrics
        """
        if not hasattr(self, 'pipe'): 
            self.load_pipeline()
        
        correct = 0
        total = len(test_queries)
        factuality_scores = []
        
        for item in test_queries:
            query = item.get('query', '')
            expected = item.get('expected_answer', '').lower()
            
            try:
                response = self.answer(query)
                answer = response.get('answer', '').lower()
                context = response.get('context', [])
                
                # Check if answer contains expected information
                answer_contains_expected = expected in answer if expected else True
                
                # Check if answer is grounded in context
                context_text = ' '.join(context).lower()
                answer_grounded = any(word in context_text for word in answer.split()[:5]) if answer else False
                
                # Factuality score: combination of correctness and grounding
                factuality = 1.0 if (answer_contains_expected and answer_grounded) else 0.5 if answer_grounded else 0.0
                factuality_scores.append(factuality)
                
                if factuality >= 0.5:
                    correct += 1
                    
            except Exception as e:
                factuality_scores.append(0.0)
        
        accuracy = correct / total if total > 0 else 0.0
        avg_factuality = sum(factuality_scores) / len(factuality_scores) if factuality_scores else 0.0
        
        return {
            'grounding_accuracy': float(accuracy),
            'average_factuality_score': float(avg_factuality),
            'total_queries': total,
            'correct_answers': correct
        }