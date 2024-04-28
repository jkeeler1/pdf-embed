import getopt
import os
import sys
import openai
from PyPDF2 import PdfReader
from pinecone import Pinecone


def main(argv):
    # file_path = "/Users/jkeeler/Documents/ai/animals/arctic-fox.pdf"
    # id = "arctic-fox
    fname, id = inputs(argv)

    # Break the pdf into chunks
    texts = process_pdf(fname)

    # Create the embeddings from the text chunks
    embeddings = create_embeddings(id, fname, texts)

    # Upsert the embeddings to Pinecone
    upsert_embeddings_to_pinecone(embeddings)
    print("Success")


def process_pdf(file_path):
    reader = PdfReader(file_path)
    texts = []
    for page in reader.pages:
        text = page.extract_text()
        texts.append(text)
    return texts

def create_embeddings(id, filename, texts):
    # Initialize OpenAI
    MODEL = "text-embedding-ada-002"
    openAIKey = os.environ.get("OPENAI_API_KEY")
    openAIOrgId = os.environ.get("OPENAI_ORG_ID")

    # initialize AI client
    openAIClient = openai.OpenAI(api_key=openAIKey, organization=openAIOrgId)
    # list of tuples where each tuple will be one chunk of the file (one row in pinecone)
    embeddings = []
    file = extractFilename(filename);

    for index, text in enumerate(texts):
        response = openAIClient.embeddings.create(
            model=MODEL,
            input=text
        )
        embedding = response.data[0].embedding
        # print(embedding)
        chunkId = id + "_" + str(index)
        embeddings.append((chunkId, embedding, {'name': id, 'filename': file, 'chunk': text}))
    return embeddings


# Define a function to upsert embeddings to Pinecone
def upsert_embeddings_to_pinecone(embeddings):

    if len(embeddings) == 0:
        return

    # Initialize Pinecone
    pineconeKey = os.environ.get("PINECONE_API_KEY")
    pineconeClient = Pinecone(api_key=pineconeKey, environment='gcp-starter')
    index = pineconeClient.Index("animals")

    # ("vec1", [0.1, 0.2, 0.3, 0.4], {"genre": "drama"}),
    response = index.upsert(
        vectors=embeddings,
        namespace="animals"
    )
    print(response)

# Define a function to create embeddings
def extractFilename(filename):
    vals = filename.split("/")
    return vals[len(vals) - 1]


def usage():
    print('usage: upsert.py -f <inputfile> -i <id>')


def inputs(argv):
    # Get the input values

    fname: str = ''
    id: str = ''

    try:
        opts, args = getopt.getopt(argv, "hf:i:", ["help", "file=", "id="])
    except getopt.GetoptError:
        print('usage: upsert.py -h')
        print('usage: upsert.py -f <inputfile> -i <id>')
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print('upsert.py -f <inputfile> -i <id>')
            sys.exit()
        elif opt in ("-f", "--file"):
            fname = arg
        elif opt in ("-i", "--id"):
            id = arg

    print('filename=', fname)
    print('id', id)

    return fname, id


if __name__ == "__main__":
    main(sys.argv[1:])
