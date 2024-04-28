# pdf-embed

### Overview

This script will take a pdf, break it into text chunks, run it through OpenAI embeddings, and store
the result in a pinecone database.

This was run with Python 3.12.2

### Usage

`python upsert.py -f /Users/jkeeler/Documents/ai/animals/arctic-fox.pdf -i arctic-fox`
where:
 -f is the pdf you wish to read
 -i is the id that should be associated with

`python upsert.py -h for help`

### Requirements
Assumes you have set the following environment variables
* PINECONE_API_KEY
* OPENAI_API_KEY
* OPENAI_ORG_ID

#### Light Reading
https://pypi.org/project/pinecone-client/
https://docs.pinecone.io/reference/api/data-plane/upsert
