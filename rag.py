import streamlit as st
import os
import tempfile
import logging
import sys

from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex

# For ngrok tunnel
from pyngrok import ngrok, conf

# Setup logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# HuggingFace + LlamaIndex cache dirs
os.environ["HF_HOME"] = "C:/Users/BasUx/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "C:/Users/BasUx/.cache/huggingface"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "C:/Users/BasUx/.cache/huggingface"
os.environ["LLAMA_INDEX_CACHE_DIR"] = "C:/Users/BasUx/.cache/huggingface"

# Init models
def init_llm():
    llm = Ollama(model="gemma2", request_timeout=600.0, temperature=0.7, streaming=True)
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.llm = llm
    Settings.embed_model = embed_model
    return llm, embed_model

llm, embed_model = init_llm()

# --- Streamlit UI ---
st.set_page_config(page_title="RAG Demo", layout="wide")
st.title("📄 RAG with Streaming (Streamlit + LlamaIndex)")

# Upload documents
uploaded_files = st.file_uploader("Upload your documents", type=["txt", "pdf"], accept_multiple_files=True)

# Save uploaded docs to temp dir
docs_dir = tempfile.mkdtemp()
if uploaded_files:
    for f in uploaded_files:
        path = os.path.join(docs_dir, f.name)
        with open(path, "wb") as out:
            out.write(f.getvalue())
    st.success(f"{len(uploaded_files)} document(s) uploaded.")

# Build index
if uploaded_files:
    documents = SimpleDirectoryReader(docs_dir).load_data()
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine(streaming=True, similarity_top_k=2)

    # User prompt input
    user_prompt = st.text_area("Enter your custom prompt:", placeholder="Ask something about your docs...")

    if st.button("Run Query") and user_prompt.strip():
        st.info("Generating response...")

        response_stream = query_engine.query(user_prompt)

        # Stream response token by token
        response_placeholder = st.empty()
        streamed_text = ""
        for token in response_stream.response_gen:
            streamed_text += token
            response_placeholder.markdown(streamed_text)
