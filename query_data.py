import os
import yaml
import logging
import argparse
from create_db import SentenceTransformerEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
from sentence_transformers import SentenceTransformer, util

# === Enable caching for development ===
set_llm_cache(SQLiteCache(database_path=".langchain_cache.db"))

# === Load Parameters (from "params.yaml") ===
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

LOG_PATH = params["LOG_PATH"]
DATA_PATH = params["DATA_PATH"]
CHROMA_DB_PATH = params["CHROMA_DB_PATH"]
EMBEDDING_MODEL_NAME = params["EMBEDDING_MODEL_NAME"]
GENERATION_MODEL_NAME = params["GENERATION_MODEL_NAME"]


# OLD_PROMPT_TEMPLATE = """
# Answer the question based only on the following context:

# {context}

# ---

# Answer the question based on the above context: {question}
# """

PROMPT_TEMPLATE = """
You must answer the question strictly based on the context provided below, avoid using any external knowledge. 
If the answer is not in the context, respond with: "The information is not available in the provided context."

Context:
{context}

---

Question: {question}
Answer:"""


# === Setup Logging ===
LOG_DIR = os.path.join(os.getcwd(), LOG_PATH)
os.makedirs(LOG_DIR, exist_ok=True)  # Create the logs directory if it doesn't exist
LOG_FILE = os.path.join(LOG_DIR, "query_data.log")

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

# Query: "How does Alice meet the Mad Hatter?"

# === Load local LLM ===
def load_local_model(model_name=GENERATION_MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    model.eval()
    return tokenizer, model

def generate_response(prompt, tokenizer, model, max_tokens=512):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id
        )    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# === Guardrail: Validate if answer is supported by context ===
semantic_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

def is_answer_supported_by_context(answer, context_chunks, threshold=0.65):
    answer_embedding = semantic_model.encode(answer, convert_to_tensor=True)
    context_embeddings = semantic_model.encode(context_chunks, convert_to_tensor=True)
    cosine_scores = util.cos_sim(answer_embedding, context_embeddings)
    max_score = cosine_scores.max().item()
    return max_score >= threshold

def main():
    logger = setup_logger("query_logger", LOG_FILE)
    logger.info("************Starting data querying workflow...*************")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    # logger.info("Query text: %s", query_text)
    
    logger.info("Preparing Chroma DB...")
    embedding_function = SentenceTransformerEmbeddings()
    db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embedding_function)
    logger.info("Loaded Chroma DB from %s", CHROMA_DB_PATH)
    
    logger.info("Performing similarity search from DB...")
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    # # Return type of search
    # List[Tuple[Document, float]]
    if len(results) == 0 or results[0][1] < 0.7:
        logger.warning("No results found with high relevance score.")
        logger.warning("Relevance score: %f", results[0][1])
        return
    logger.info("Found %d results with relevance scores.", len(results))
    
    logger.info("Formatting results...")
    context_chunks = [doc.page_content for doc, _score in results]
    context_text = "\n\n---\n\n".join(context_chunks)
    # logger.info("Context text: %s", context_text)
    
    logger.info("Creating prompt...")
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # logger.info("Prompt: %s", prompt)
    
    logger.info("Loading local LLM...")
    tokenizer, model = load_local_model(GENERATION_MODEL_NAME)

    logger.info("Generating response from local model...")
    response_text = generate_response(prompt, tokenizer, model)
    # logger.info("Response: %s", response_text)
    # logger.info("Response length: %d", len(response_text))
    
    # === Apply semantic similarity guardrail ===
    logger.info("Validating answer relevance using semantic similarity...")
    if not is_answer_supported_by_context(response_text, context_chunks):
        logger.warning("Answer rejected by semantic guardrail.")
        response_text = "The information is not available in the provided context."
    
    logger.info("Extracting sources...")
    sources = [doc.metadata.get("source", None) for doc, _score in results]
    # logger.info("Sources: %s", sources)
    
    logger.info("Formatting final response...")
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    logger.info("Formatted response: %s", formatted_response)
    print(formatted_response)
    
    logger.info("***********Data querying workflow completed successfully.*********")


if __name__ == "__main__":
    main()
