from langchain_community.document_loaders import TextLoader,DirectoryLoader, PyMuPDFLoader , WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv, find_dotenv
from langchain_core.documents import Document
import fitz
import json


load_dotenv(find_dotenv())

BOOKS_DATA_PATH = "data/Books/"
SCRAPTED_DATA_PATH = "data/scraped_data_gfg/"


def load_pdf_files(data):
    loader = DirectoryLoader(data,glob='*.pdf',loader_cls=PyMuPDFLoader)
    documents = loader.load()

    return documents

pdf_documents = load_pdf_files(BOOKS_DATA_PATH)
print("PDF Docs Loaded Successfully with Length : ",len(pdf_documents))



def load_scraped_data(data):
    documents = []

    for filename in os.listdir(data):
        if filename.lower().endswith('.json'):
            filepath = os.path.join(data,filename)
            with open(filepath,'r',encoding='utf-8') as f:
                data_content = json.load(f)
            

            page_content = json.dumps(data_content,ensure_ascii=False)
            metadata = {'source': filepath, 'file_type': 'json', 'filename': filename}

            documents.append(Document(page_content=page_content,metadata=metadata))
    
    return documents

json_documents = load_scraped_data(SCRAPTED_DATA_PATH)
print("JSON Docs Loaded Successfully with Length : ",len(json_documents))
combined_documents = pdf_documents + json_documents
print("Documents Loaded Successfully with Length : ",len(combined_documents))

def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800,chunk_overlap=150)
    text_chunks = text_splitter.split_documents(extracted_data)

    return text_chunks

text_chunks = create_chunks(extracted_data=combined_documents)
print("Combined Documents Converted Successfully into Chunks with Length : ",len(text_chunks))

def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(model_name = "BAAI/bge-small-en")

    return embedding_model

embedding_model = get_embedding_model()

DB_FAISS_PATH = "vectorstore/db_faiss"


print(f"\nStarting to create FAISS vector store from {len(text_chunks)} chunks.")
print("This step involves generating embeddings for all chunks and can take a long time...")
db = FAISS.from_documents(text_chunks,embedding_model)
print("FAISS vector store created successfully!")

# --- Step 3: Save the FAISS vector store ---
print(f"\nSaving FAISS vector store locally to: {DB_FAISS_PATH}")
db.save_local(DB_FAISS_PATH)
print("Saved Successfully!")

print("\n--- Data Preparation Process Complete ---")