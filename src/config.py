class Config:
    LLM_MODEL_ID = 'mistralai/Mistral-7B-Instruct-v0.3'
    DATA_PATH = "data/marketing_instruct.json"
    EMBEDDING_MODEL = "mixedbread-ai/mxbai-embed-large-v1"
    INDEX_PATH = "models/faiss_index_gpu"
    METADATA_STORE_PATH = "models/metadata_store.parquet"
    ADAPTER_PATH = "models/final_adapter"