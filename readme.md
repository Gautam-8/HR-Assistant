# 🤖 HR Assistant — LangChain RAG with LM Studio

This is an AI-powered HR Knowledge Assistant that allows employees to ask HR-related questions like:

- "How many vacation days do I get?"
- "Can I work remotely during probation?"
- "What benefits do we get for health insurance?"

The app uses **LangChain**, **Chroma**, and a **local LLM (via LM Studio)** to retrieve answers from uploaded HR documents.

---

## 🔧 Features

✅ Upload multi-format HR documents (PDF, DOCX, TXT)  
✅ Chunk, embed, and store content using local embedding model  
✅ Query through a Streamlit UI  
✅ Built-in RAG Chain with LangChain for document-grounded answers  
✅ Sources shown with each answer

---

## 🛠 Tech Stack

- `LangChain` + `Chroma`
- `LM Studio` for LLM + embeddings (OpenAI-compatible API)
- `Streamlit` for the user interface

---

## 🚀 How to Run

### 1. Install requirements

```bash
pip install -r requirements.txt
```
