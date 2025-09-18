# simple-rag
## **Pre-requisites**
- Download Ollama for Windows: https://ollama.com/download/OllamaSetup.exe
- Verify installation: `ollama --version` in cmd
- Download LLM: `ollama run gemma2`
  
## **How to Run**
1. Download repo
2. Open in VSCode
3. Create venv
- `python -m venv venv`
- `venv\Scripts\activate`

4. Install deps
- `pip install -r requirements.txt`

5. Start streamlit server
- `streamlit run app.py`
