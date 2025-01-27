# imports
import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

from dotenv import load_dotenv
import gradio as gr

# model and db initilaised
MODEL = 'gpt-4o-mini'

#load environment cars
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

# LangChain for RAG conversation chain
llm  = ChatOpenAI(model=MODEL,temperature=0.7)

memory = ConversationBufferMemory(memory_key='chat_history', return_messages= True)

#loading existing vectorstore
embeddings = OpenAIEmbeddings(model = 'text-embedding-3-small')  # default is text embedding ada-0002
DB_NAME = 'vector-db'
vectorstore = Chroma(persist_directory= DB_NAME, embedding_function= embeddings)
retriever = vectorstore.as_retriever()

# setting up the chain
conversation_chain = ConversationalRetrievalChain.from_llm(llm = llm,retriever = retriever, memory = memory)

# query = "Can you describe Insurellm in a few sentences"
# result = conversation_chain.invoke({"question":query})
# print(result['answer'])

# gradio interface
def chat(message,history):
    result = conversation_chain.invoke({"question":message})
    return result['answer']

view = gr.ChatInterface(chat).launch()