import numpy as np
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from sklearn.manifold import TSNE
import plotly.graph_objects as go

#loading existing vectorstore
embeddings = OpenAIEmbeddings(model = 'text-embedding-3-small')  # default is text embedding ada-0002
DB_NAME = 'vector-db'
vectorstore = Chroma(persist_directory= DB_NAME, embedding_function= embeddings)
collection = vectorstore._collection

# Visualize Prework
result = collection.get(include = ['embeddings','documents','metadatas'])
vectors = np.array(result['embeddings'])
documents = result['documents']
doc_types = [metadata['doc_type'] for metadata in result['metadatas']]
doc_to_color = dict(zip(['products', 'employees', 'contracts', 'company'],['blue', 'green', 'red', 'orange']))
colors = [doc_to_color[t] for t in doc_types]

# dimensionality reduction: transforming to 3d
tsne = TSNE(n_components=3, random_state=42)
reduced_vectors = tsne.fit_transform(vectors)

#create 3d scatterplot
# Create the 3D scatter plot
fig = go.Figure(data=[go.Scatter3d(
    x=reduced_vectors[:, 0],
    y=reduced_vectors[:, 1],
    z=reduced_vectors[:, 2],
    mode='markers',
    marker=dict(size=5, color=colors, opacity=0.8),
    text=[f"Type: {t}<br>Text: {d[:100]}..." for t, d in zip(doc_types, documents)],
    hoverinfo='text'
)])

fig.update_layout(
    title='3D Chroma Vector Store Visualization',
    scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='z'),
    width=900,
    height=700,
    margin=dict(r=20, b=10, l=10, t=40)
)

fig.show()
