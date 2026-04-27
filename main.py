import os
import sys
import glob
import logging
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s'
)
logger = logging.getLogger(__name__)

def initialize_environment():
    """Load environment variables and check for API keys."""
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        logger.error(
            "GOOGLE_API_KEY environment variable is not set. "
            "Please create a .env file and add your API key."
        )
        sys.exit(1)

def find_pdf_file(docs_dir="docs"):
    """Finds a PDF file in the specified directory."""
    if not os.path.exists(docs_dir):
        logger.error(f"Directory '{docs_dir}' does not exist.")
        os.makedirs(docs_dir)
        logger.info(f"Created '{docs_dir}' directory. Please place a PDF file inside it and run again.")
        sys.exit(1)
    
    pdf_files = glob.glob(os.path.join(docs_dir, "*.pdf"))
    if not pdf_files:
        logger.error(f"No PDF files found in '{docs_dir}' directory. Please add a PDF and run again.")
        sys.exit(1)
        
    pdf_path = pdf_files[0]
    logger.info(f"Found PDF file to process: {os.path.basename(pdf_path)}")
    return pdf_path

def build_vector_store(pdf_path):
    """Loads PDF, splits into chunks, and creates FAISS vector store."""
    try:
        logger.info(f"Loading document: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
    except Exception as e:
        logger.error(f"Failed to load PDF: {e}")
        sys.exit(1)

    logger.info("Splitting document into chunks...")
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    texts = text_splitter.split_documents(documents)
    logger.info(f"Created {len(texts)} chunks from the document.")

    logger.info("Generating embeddings and storing in FAISS vector database...")
    try:
        # Using Google Generative AI embeddings
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        vector_store = FAISS.from_documents(texts, embeddings)
        logger.info("Vector database successfully initialized.")
        return vector_store
    except Exception as e:
        logger.error(f"Failed to create embeddings or vector store: {e}")
        sys.exit(1)

def build_qa_chain(vector_store):
    """Initializes the RetrievalQA chain with Gemini Pro."""
    logger.info("Initializing Gemini LLM and QA chain...")
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.3,
            convert_system_message_to_human=True
        )
        retriever = vector_store.as_retriever()
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )

        return qa_chain
    except Exception as e:
        logger.error(f"Failed to initialize QA chain: {e}")
        sys.exit(1)

def chat_loop(qa_chain):
    """Runs the continuous CLI loop for user to query the document."""
    print("\n" + "="*60)
    print("LangChain PDF Chatbot Initialized")
    print("Type your questions below. Type 'exit' or 'quit' to terminate.")
    print("="*60)
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            # Empty input check
            if not user_input:
                logger.warning("Empty input received. Please enter a question.")
                continue
                
            # Exit check
            if user_input.lower() in ['exit', 'quit']:
                print("\nExiting application. Goodbye!")
                break
                
            # Query the RAG chain
            logger.info("Querying the model...")
            response = qa_chain.invoke(user_input)
            
            # Print the result depending on the Langchain version format
            answer = response.get('result') or response.get('answer', 'No answer generated.')
            print(f"\nBot: {answer}\n")
            
            # Print source documents chunks (Bonus)
            source_docs = response.get("context", [])
            if source_docs:
                print("--- Source Chunks Used ---")
                for i, doc in enumerate(source_docs, 1):
                    # Trim content for display purposes
                    content_preview = doc.page_content[:150].replace('\n', ' ')
                    page_num = doc.metadata.get('page', 'Unknown')
                    print(f"[{i}] Page {page_num}: {content_preview}...")
                print("-" * 26)
            
        except KeyboardInterrupt:
            print("\n\nExiting application. Goodbye!")
            break
        except Exception as e:
            logger.error(f"An error occurred while generating a response: {e}")

def main():
    initialize_environment()
    
    pdf_path = find_pdf_file("docs")
    vector_store = build_vector_store(pdf_path)
    qa_chain = build_qa_chain(vector_store)
    
    chat_loop(qa_chain)

if __name__ == "__main__":
    main()
