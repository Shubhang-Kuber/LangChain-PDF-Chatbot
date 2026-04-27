# 📄 LangChain PDF Chatbot with Google Gemini

A robust, command-line interface (CLI) application built with Python and **LangChain**. It allows users to interactively chat with their local PDF documents. The chatbot leverages a **Retrieval-Augmented Generation (RAG)** approach using Google's **Gemini** models and a local **FAISS** vector database to provide accurate, context-aware answers directly from the provided document.

---

## ✨ Features

- **Document Ingestion:** Seamlessly loads local PDF files from the `docs/` directory using `PyPDFLoader`.
- **Intelligent Chunking:** Splits large PDF documents into manageable segments with `CharacterTextSplitter`.
- **Advanced Embeddings:** Leverages Google's `gemini-embedding-001` for accurate, high-quality vector representations.
- **Local Vector Database:** Uses `FAISS` to store and instantly retrieve the most relevant text chunks.
- **Conversational RAG:** Powers the core response engine with the latest **Gemini 2.5 Flash** model for fast, intelligent reasoning.
- **Transparent Sourcing:** Prints the exact document chunks and page metadata used to construct the answer.
- **Error Resilient:** Cleanly handles missing files, empty inputs, API key retrieval, and graceful exits.

---

## 🛠️ Prerequisites

- **Python 3.10+** installed on your system.
- A **Google Gemini API Key**. You can get one for free at [Google AI Studio](https://aistudio.google.com/).

---

## 🚀 Installation & Setup

**1. Clone the Repository:**
```bash
git clone https://github.com/your-username/LangChain-PDF-Chatbot.git
cd LangChain-PDF-Chatbot
```

**2. Set up a Virtual Environment (Recommended):**
```bash
python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

**3. Install Dependencies:**
```bash
pip install -r requirements.txt
```

**4. Configure your API Key:**
* Create a `.env` file in the root directory of the project.
* Add your Google API key to the file like this:
```env
GOOGLE_API_KEY="AIzaSyYourVerySecretGoogleKeyHere"
```
*(Note: Never share this key or commit your `.env` file to a public repository!)*

---

## 💡 Usage

**1. Add your PDF:**
Place a single PDF file you want to chat with inside the `docs/` folder (the application creates this folder automatically if it's missing, but will prompt you to add a file).

**2. Run the Chatbot:**
```bash
python main.py
```

**3. Chat!**
Wait a few seconds for the document to be processed, chunked, and embedded. Once you see the `You:` prompt, start asking questions about the document's content.

Type `exit` or `quit` when you are done.

---

## 🗂️ Project Structure

```text
├── docs/                # Place your target PDF file here
├── .env                 # (Ignored) Your localized environment variables/API Keys
├── .gitignore           # Standard python & security exclusions
├── main.py              # Main application logic & CLI loop
├── requirements.txt     # Python package dependencies
└── README.md            # Project documentation
```

---

> **Note:** This project relies on the free tier of the Google Gemini API. If you submit questions very quickly in rapid succession, you might temporarily hit rate limits (`429 Too Many Requests`). Just wait a minute and try again.