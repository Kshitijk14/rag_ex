import os
import shutil
import yaml
import glob
import traceback
import logging
import time
import psutil
# from huggingface_hub import login
# from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer


# === Load Environment Variables (from ".env") ===
# load_dotenv()
# hf_token=os.environ['HUGGINGFACEHUB_API_TOKEN']
# login(hf_token)

# === Load Parameters (from "params.yaml") ===
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

LOG_PATH = params["LOG_PATH"]
DATA_PATH = params["DATA_PATH"]
CHROMA_DB_PATH = params["CHROMA_DB_PATH"]
CHROMA_DB_FILE = params["CHROMA_DB_FILE"]
EMBEDDING_MODEL_NAME = params["EMBEDDING_MODEL_NAME"]
CHUNK_SIZE = params["CHUNK_SIZE"]
CHUNK_OVERLAP = params["CHUNK_OVERLAP"]


# === Setup Logging ===
LOG_DIR = os.path.join(os.getcwd(), LOG_PATH)
os.makedirs(LOG_DIR, exist_ok=True)  # Create the logs directory if it doesn't exist
LOG_FILE = os.path.join(LOG_DIR, "create_db.log")

def setup_logger(name, log_file):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Prevent adding multiple handlers on re-imports
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(message)s",
#     handlers=[
#         logging.FileHandler(LOG_FILE),
#         logging.StreamHandler()
#     ]
# )


# === Embedding Class ===
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name=EMBEDDING_MODEL_NAME, logger=None):
        try:
            self.logger = logger or logging.getLogger(__name__)
            self.model = SentenceTransformer(model_name)
            self.logger.info("Loaded SentenceTransformer model: %s", model_name)
        except Exception as e:
            self.logger.error("Failed to load embedding model: %s", e)
            self.logger.debug(traceback.format_exc())
            exit(1)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        try:
            return self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True).tolist()
        except Exception as e:
            self.logger.error("Embedding documents failed: %s", e)
            self.logger.debug(traceback.format_exc())
            return []

    def embed_query(self, text: str) -> list[float]:
        try:
            return self.model.encode([text], convert_to_numpy=True)[0].tolist()
        except Exception as e:
            self.logger.error("Embedding query failed: %s", e)
            self.logger.debug(traceback.format_exc())
            return []

# === Load Documents ===
def load_documents(logger):
    """Load .md documents from the specified path."""
    try:
        filepaths = glob.glob(os.path.join(DATA_PATH, "*.md"))
        documents = []
        if not filepaths:
            logger.warning("No .md files found in path: %s", DATA_PATH)
        
        # loader = TextLoader(DATA_PATH, glob="*.md")
        # documents = loader.load()
        
        for path in filepaths:
            loader = TextLoader(path, encoding='utf-8')
            documents.extend(loader.load())

        logger.info("Loaded %d documents from %s", len(documents), DATA_PATH)
        return documents
    except Exception as e:
        logger.error("Failed to load documents: %s", e)
        logger.debug(traceback.format_exc())
        return []

# === Split Documents ===
def split_text(documents, logger):
    """Split documents into smaller chunks."""
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            add_start_index=True,
        )
        chunks = text_splitter.split_documents(documents)
        logger.info("Split %d documents into %d chunks.", len(documents), len(chunks))

        # documents = chunks[10]
        # print(f"Chunk text: {documents.page_content}")
        # print(f"Chunk metadata: {documents.metadata}")
        
        # if chunks:
        #     logger.debug("Sample chunk: %s", chunks[10].page_content)
            
        #     logger.info(f"Sample chunk text: {chunks[10].page_content}")
        #     logger.info(f"Sample chunk metadata: {chunks[10].metadata}")

        return chunks
    except Exception as e:
        logger.error("Failed to split text: %s", e)
        logger.debug(traceback.format_exc())
        return []


# Helper: check if any process is locking the file
def is_file_in_use(filepath):
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            for item in proc.open_files():
                if filepath in item.path:
                    return True
        except Exception:
            continue
    return False

# === Save to Chroma ===
def save_to_chromadb(chunks: list[Document], logger):
    """Create a Chroma database from the chunks."""
    try:
        embeddings = SentenceTransformerEmbeddings(logger=logger)

        # Clear the existing database if it exists
        if os.path.exists(CHROMA_DB_PATH):
            chroma_file = os.path.join(CHROMA_DB_PATH, CHROMA_DB_FILE)
            
            # Wait until file is not locked
            for _ in range(5):
                if not is_file_in_use(chroma_file):
                    break
                logger.warning("ChromaDB file is in use, retrying...")
                time.sleep(1)
            else:
                raise RuntimeError("ChromaDB is locked by another process.")
            
            shutil.rmtree(CHROMA_DB_PATH)
            logger.info("Cleared existing Chroma database at %s", CHROMA_DB_PATH)

        db = Chroma.from_documents(
            chunks, 
            embeddings, 
            persist_directory=CHROMA_DB_PATH
        )
        
        logger.info("Saved %d chunks to Chroma database at %s", len(chunks), CHROMA_DB_PATH)
    except Exception as e:
        logger.error("Failed to save to ChromaDB: %s", e)
        logger.debug(traceback.format_exc())

# === Data Store Workflow ===
def generate_data_store(logger):
    """Generate a data store from the documents."""
    try:
        documents = load_documents(logger)
        if not documents:
            logger.error("No documents to process. Exiting.")
            return False
        
        chunks = split_text(documents, logger)
        if not chunks:
            logger.error("No chunks generated. Exiting.")
            return False
        
        save_to_chromadb(chunks, logger)
        logger.info("Database creation workflow completed successfully.")
        return True
    
    except Exception as e:
        logger.error("Unexpected error in generate_data_store(): %s", e)
        logger.debug(traceback.format_exc())
        return False


# === Main Execution ===
def main():
    logger = setup_logger("create_db_logger", LOG_FILE)
    logger.info("*********************Starting database creation workflow...*****************")
    pipeline = generate_data_store(logger)
    
    if not pipeline:
        logger.warning("Workflow ended with errors.")
    else:
        logger.info("***************Workflow completed successfully.****************")


if __name__ == "__main__":
    main()