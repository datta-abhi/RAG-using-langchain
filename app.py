import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import List, Tuple
from dotenv import load_dotenv
import gradio as gr

# Load environment variables
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

# Model and database initialization
MODEL = 'gpt-4o-mini'
DB_NAME = 'vector-db'

# Initialize LangChain components
llm = ChatOpenAI(model=MODEL, temperature=0.7)
embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
retriever = vectorstore.as_retriever()

# Prompts
contextualize_q_system_prompt = """
Given a chat history and the latest user question which might reference context in the chat history, 
formulate a standalone question which can be understood without the chat history. 
Do NOT answer the question, just reformulate it if needed and otherwise return it as is.
"""

qa_system_prompt = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context 
to answer the question. If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.

{context}
"""

# Create prompt templates
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Create chains
history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

def format_chat_history(chat_history: List[Tuple[str, str]]) -> List:
    """Convert Gradio chat history to LangChain message format"""
    formatted_history = []
    for user_msg, assistant_msg in chat_history:
        formatted_history.append(HumanMessage(content=user_msg))
        if assistant_msg:  # Only add assistant message if it exists
            formatted_history.append(AIMessage(content=assistant_msg))
    return formatted_history

def chat_response(message: str, history: List[Tuple[str, str]]) -> str:
    """Process a single message and return the response"""
    formatted_history = format_chat_history(history)
    
    try:
        # Process the user's query through the retrieval chain
        result = rag_chain.invoke({
            "input": message,
            "chat_history": formatted_history
        })
        return result["answer"]
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Create and launch the Gradio interface
demo = gr.ChatInterface(
    fn=chat_response,
    title="RAG-powered Chatbot",
    description="Ask questions about your documents",
    examples=["Can you describe Insurellm in a few sentences?"],
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    demo.launch(share=False, inbrowser=True)
