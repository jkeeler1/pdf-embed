import getopt
import os
import re
import sys
import openai
from PyPDF2 import PdfReader
from pinecone import Pinecone


def usage():
    print('usage: upset.py -f <inputfile> -i <id>')


def inputs(argv):
    # Get the input values

    fname = ''
    id = ''

    try:
        opts, args = getopt.getopt(argv, "hf:i:", ["help", "file=", "id="])
    except getopt.GetoptError:
        print('usage: upset.py -h')
        print('usage: upset.py -f <inputfile> -i <id>')
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print('upset.py -f <inputfile> -i <id>')
            sys.exit()
        elif opt in ("-f", "--file"):
            fname = arg
        elif opt in ("-i", "--id"):
            id = arg

    print ('filename=', fname)
    print ('id', id)

    return fname, id

def main(argv):
    # file_path = "/Users/jkeeler/Documents/ai/animals/arctic-fox.pdf"
    fname, id = inputs(argv)

    # Process a PDF and create embeddings
    texts = process_pdf(fname)

    textToEmbedding = create_embeddings(texts)

    # Upsert the embeddings to Pinecone
    upsert_embeddings_to_pinecone(id, textToEmbedding, fname)
    print ("Success")

# Define a function to preprocess text
def preprocess_text(text):
    # Replace consecutive spaces, newlines and tabs
    text = re.sub(r'\s+', ' ', text)
    return text

def process_pdf(file_path):
    reader = PdfReader(file_path)
    texts = []
    for page in reader.pages:
        text = page.extract_text()
        texts.append(text)
    return texts

# Define a function to create embeddings
def create_embeddings(texts):
    # Initialize OpenAI
    MODEL = "text-embedding-ada-002"
    openAIKey = os.environ.get("OPENAI_API_KEY")
    openAIOrgId = os.environ.get("OPENAI_ORG_ID")

    # initialize AI client
    openAIClient = openai.OpenAI(api_key=openAIKey, organization=openAIOrgId)
    textToEmbedding = {}

    for text in texts:
        response = openAIClient.embeddings.create(
            model=MODEL,
            input=text
        )
        embedding = response.data[0].embedding
        print(embedding)
        textToEmbedding[text] = embedding
    return textToEmbedding

# Define a function to upsert embeddings to Pinecone
def upsert_embeddings_to_pinecone(id, textToEmbedding, filename):
    # Initialize Pinecone
    pineconeKey = os.environ.get("PINECONE_API_KEY")
    pineconeClient = Pinecone(api_key=pineconeKey, environment='gcp-starter')
    index = pineconeClient.Index("animals")

    for fileText, embedding in textToEmbedding.items():
        print("upserting vector")
        response = index.upsert(vectors=[
                {
                    'id':id,
                    'values':embedding,
                    'metadata':{'name':id, 'filename':filename, 'section':fileText}
                }])
        print(response)

if __name__ == "__main__":
   main(sys.argv[1:])