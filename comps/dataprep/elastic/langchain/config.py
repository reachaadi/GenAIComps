import os

ES_CONNECTION_STRING = os.getenv("ES_CONNECTION_STRING", "localhost")

# Embedding model
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-base-en-v1.5")

# TEI Embedding endpoints
TEI_EMBEDDING_ENDPOINT = os.getenv("TEI_EMBEDDING_ENDPOINT", "")

# Vector Index Configuration
INDEX_NAME = os.getenv("INDEX_NAME", "rag-elastic")

# chunk parameters
CHUNK_SIZE = os.getenv("CHUNK_SIZE", 1500)
CHUNK_OVERLAP = os.getenv("CHUNK_OVERLAP", 100)

current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
