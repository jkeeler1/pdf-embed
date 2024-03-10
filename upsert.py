import os
import re
# import pdfplumber
import openai
from pinecone import Pinecone
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Initialize OpenAI

MODEL = "text-embedding-ada-002"
openAIKey = os.environ.get("OPENAI_API_KEY")
openAIOrgId = os.environ.get("OPENAI_ORG_ID")
pineconeKey = os.environ.get("PINECONE_API_KEY")

#initialize AI client
openAIClient = openai.OpenAI(api_key=openAIKey, organization=openAIOrgId)

# Initialize Pinecone
pineconeClient = Pinecone(api_key=pineconeKey, environment='gcp-starter')
index = pineconeClient.Index("animals")

# Define a function to preprocess text
def preprocess_text(text):
    # Replace consecutive spaces, newlines and tabs
    text = re.sub(r'\s+', ' ', text)
    return text

def process_pdf(file_path):
    # create a loader
    loader = PyPDFLoader(file_path)
    # load your data
    data = loader.load()
    # Split your data up into smaller documents with Chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.split_documents(data)
    # Convert Document objects into strings
    texts = [str(doc) for doc in documents]
    return texts

# Define a function to create embeddings
def create_embeddings(texts):
    embeddings_list = []
    for text in texts:
        res = openAIClient.embeddings.create(input=[text], model=MODEL)
        embeddings_list.append(res['data'][0]['embedding'])
    return embeddings_list

# Define a function to upsert embeddings to Pinecone
def upsert_embeddings_to_pinecone(index, embeddings, ids):
    index.upsert(vectors=[(id, embedding) for id, embedding in zip(ids, embeddings)])

# Process a PDF and create embeddings
file_path = "/Users/jkeeler/Documents/ai/animals/arctic-fox.pdf"
texts = process_pdf(file_path)
embeddings = create_embeddings(texts)

# Upsert the embeddings to Pinecone
# upsert_embeddin`gs_to_pinecone(index, embeddings, [file_path])