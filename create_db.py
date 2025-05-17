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
from langchain_community.vectorstores import Chroma
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
MODEL_NAME = params["MODEL_NAME"]
CHUNK_SIZE = params["CHUNK_SIZE"]
CHUNK_OVERLAP = params["CHUNK_OVERLAP"]


# === Setup Logging ===
LOG_DIR = os.path.join(os.getcwd(), LOG_PATH)
os.makedirs(LOG_DIR, exist_ok=True)  # Create the logs directory if it doesn't exist
LOG_FILE = os.path.join(LOG_DIR, "create_db.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)


# === Embedding Class ===
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name=MODEL_NAME):
        try:
            self.model = SentenceTransformer(model_name)
            logging.info("Loaded SentenceTransformer model: %s", model_name)
        except Exception as e:
            logging.error("Failed to load embedding model: %s", e)
            logging.debug(traceback.format_exc())
            exit(1)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        try:
            return self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True).tolist()
        except Exception as e:
            logging.error("Embedding documents failed: %s", e)
            logging.debug(traceback.format_exc())
            return []

    def embed_query(self, text: str) -> list[float]:
        try:
            return self.model.encode([text], convert_to_numpy=True)[0].tolist()
        except Exception as e:
            logging.error("Embedding query failed: %s", e)
            logging.debug(traceback.format_exc())
            return []

# === Load Documents ===
def load_documents():
    """Load .md documents from the specified path."""
    try:
        filepaths = glob.glob(os.path.join(DATA_PATH, "*.md"))
        documents = []
        if not filepaths:
            logging.warning("No .md files found in path: %s", DATA_PATH)
        
        # loader = TextLoader(DATA_PATH, glob="*.md")
        # documents = loader.load()
        
        for path in filepaths:
            loader = TextLoader(path, encoding='utf-8')
            documents.extend(loader.load())

        logging.info("Loaded %d documents from %s", len(documents), DATA_PATH)
        return documents
    except Exception as e:
        logging.error("Failed to load documents: %s", e)
        logging.debug(traceback.format_exc())
        return []

# === Split Documents ===
def split_text(documents):
    """Split documents into smaller chunks."""
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            add_start_index=True,
        )
        chunks = text_splitter.split_documents(documents)
        logging.info("Split %d documents into %d chunks.", len(documents), len(chunks))

        # documents = chunks[10]
        # print(f"Chunk text: {documents.page_content}")
        # print(f"Chunk metadata: {documents.metadata}")
        
        if chunks:
            logging.debug("Sample chunk: %s", chunks[10].page_content)
            
            logging.info(f"Sample chunk text: {chunks[10].page_content}")
            logging.info(f"Sample chunk metadata: {chunks[10].metadata}")

        return chunks
    except Exception as e:
        logging.error("Failed to split text: %s", e)
        logging.debug(traceback.format_exc())
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
def save_to_chromadb(chunks: list[Document]):
    """Create a Chroma database from the chunks."""
    try:
        embeddings = SentenceTransformerEmbeddings()

        # Clear the existing database if it exists
        if os.path.exists(CHROMA_DB_PATH):
            chroma_file = os.path.join(CHROMA_DB_PATH, CHROMA_DB_FILE)
            
            # Wait until file is not locked
            for _ in range(5):
                if not is_file_in_use(chroma_file):
                    break
                logging.warning("ChromaDB file is in use, retrying...")
                time.sleep(1)
            else:
                raise RuntimeError("ChromaDB is locked by another process.")
            
            shutil.rmtree(CHROMA_DB_PATH)
            logging.info("Cleared existing Chroma database at %s", CHROMA_DB_PATH)

        db = Chroma.from_documents(
            chunks, 
            embeddings, 
            persist_directory=CHROMA_DB_PATH
        )
        
        logging.info("Saved %d chunks to Chroma database at %s", len(chunks), CHROMA_DB_PATH)
    except Exception as e:
        logging.error("Failed to save to ChromaDB: %s", e)
        logging.debug(traceback.format_exc())

# === Data Store Workflow ===
def generate_data_store():
    """Generate a data store from the documents."""
    try:
        documents = load_documents()
        if not documents:
            logging.error("No documents to process. Exiting.")
            return False
        
        chunks = split_text(documents)
        if not chunks:
            logging.error("No chunks generated. Exiting.")
            return False
        
        save_to_chromadb(chunks)
        logging.info("Database creation workflow completed successfully.")
        return True
    
    except Exception as e:
        logging.error("Unexpected error in generate_data_store(): %s", e)
        logging.debug(traceback.format_exc())
        return False


# === Main Execution ===
def main():
    logging.info("Starting database creation workflow...")
    save_to_db = generate_data_store()
    
    if not save_to_db:
        logging.warning("Workflow ended with errors.")
    else:
        logging.info("Workflow completed successfully.")


if __name__ == "__main__":
    main()