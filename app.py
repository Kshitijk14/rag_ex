import os
# To fix PyTorch x Streamlit file watcher incompatibility
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

import streamlit as st
import logging
import yaml
from query_data import (
    setup_logger, load_local_model, generate_response, is_answer_supported_by_context
)
from create_db import SentenceTransformerEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache

# === Enable LangChain cache for dev speed ===
set_llm_cache(SQLiteCache(database_path=".langchain_cache.db"))

# === Load parameters ===
try:
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
except Exception as e:
    raise RuntimeError(f"Failed to load parameters from YAML: {e}")

LOG_PATH = params["LOG_PATH"]
CHROMA_DB_PATH = params["CHROMA_DB_PATH"]
GENERATION_MODEL_NAME = params["GENERATION_MODEL_NAME"]

PROMPT_TEMPLATE = """
You must answer the question strictly based on the context provided below, avoid using any external knowledge. 
If the answer is not in the context, respond with: "The information is not available in the provided context."

Context:
{context}

---

Question: {question}
Answer:"""

# === Setup logging ===
LOG_DIR = os.path.join(os.getcwd(), LOG_PATH)
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "streamlit_app.log")
logger = setup_logger("streamlit_logger", LOG_FILE)

# === Streamlit Interface ===
st.set_page_config(page_title="RAG Query App", layout="centered")
st.title("🔍 RAG-based Query Assistant")

query = st.text_input("Enter your query", "")

if query:
    try:
        st.info("⏳ Searching knowledge base and generating response...")

        # Initialize vector DB
        embedding_function = SentenceTransformerEmbeddings()
        db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embedding_function)

        results = db.similarity_search_with_relevance_scores(query, k=3)

        if len(results) == 0 or results[0][1] < 0.7:
            st.warning("⚠️ Unable to find matching results.")
            logger.warning("Low relevance score or no results.")
        else:
            # Prepare context
            context_chunks = [doc.page_content for doc, _ in results]
            context_text = "\n\n---\n\n".join(context_chunks)

            prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(
                context=context_text,
                question=query
            )

            tokenizer, model = load_local_model(GENERATION_MODEL_NAME)
            answer = generate_response(prompt, tokenizer, model)

            # Guardrails
            if not is_answer_supported_by_context(answer, context_chunks):
                logger.warning("Answer failed semantic guardrail.")
                answer = "The information is not available in the provided context."

            # Final Display
            st.success("✅ Response Generated:")
            st.markdown(f"**Answer:** {answer}")

            sources = [doc.metadata.get("source", "Unknown") for doc, _ in results]
            st.markdown("**Sources:**")
            for i, src in enumerate(sources):
                st.markdown(f"- Source {i+1}: `{src}`")

    except Exception as e:
        st.error("❌ An error occurred while processing your query.")
        logger.exception("Unhandled exception in query pipeline: %s", str(e))
