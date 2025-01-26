#imports
import os
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from docs import chunks

# initialise openai embeddings instance
embeddings = OpenAIEmbeddings(model = 'text-embedding-3-small')

# create new Chroma datastore (if already exists then delete collection to start from scratch)
DB_NAME = 'vector-db'
if os.path.exists(DB_NAME):
    Chroma(persist_directory= DB_NAME, embedding_function= embeddings).delete_collection()
    
vectorstore = Chroma.from_documents(documents= chunks, embedding= embeddings,persist_directory= DB_NAME)
print(f"Vectorstore created with {vectorstore._collection.count()} documents")

# Fetch a sample vector and how many dimensions it has
collection = vectorstore._collection
sample_embedding = collection.get(limit = 1, include = ['embeddings'])['embeddings'][0]
dimensions = len(sample_embedding)
print("Nos of dims: ",dimensions)   
