import os
import glob
from langchain.document_loaders import DirectoryLoader,TextLoader
from langchain.text_splitter import CharacterTextSplitter

# reading documents using Langchain loaders
#taking everything in sub-folders of our knowledge-base

folders = glob.glob("knowledge-base/*")  # list of sub-folder paths
documents = []

for folder in folders:
    doc_type = os.path.basename(folder)  # extracts company if folder is knowledge-base/company
    
    #loading using Documentloader, and parsed using TextLoader
    # specifies that only markdown files should be loaded, 
    # double ** in glob to resursively search sub-directories for md files
    loader = DirectoryLoader(folder,glob = '**/*.md',loader_cls = TextLoader,
                             loader_kwargs={'autodetect_encoding': True}) 
    folder_docs = loader.load()  # list of document objects having metadata, and content
    for doc in folder_docs:
        doc.metadata['doc_type'] = doc_type  # subfolder name is saved as metadata
        documents.append(doc)

# splitting text into chunks
text_splitter = CharacterTextSplitter(chunk_size = 800, chunk_overlap = 200)
chunks = text_splitter.split_documents(documents)
# print(len(chunks))
# print(chunks[1])

# checking if all doc-types are captured in our metadata
doc_types = set(chunk.metadata['doc_type'] for chunk in chunks)  #set comprehension
# print("Document types in chunks: ",doc_types)

# checking which chunks has particular words
def check_word_in_chunk(word):
    for chunk in chunks:
        if word.lower() in chunk.page_content.lower():
            print(chunk)
            print("--"*50)
            
# check_word_in_chunk("scientist")            