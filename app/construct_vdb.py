import pandas as pd
from utils import *
from data_handling import *
from pinecone import Pinecone, ServerlessSpec
import os
from langchain.schema import Document
from langchain_community.embeddings import CohereEmbeddings
from langchain_community.vectorstores import  Pinecone as Pinecone_Langchain
import random

#1 Get the pinecone secret key and index name from the environment variables
os.environ['PINECONE_API_KEY'] = pinecone_secret_key
os.environ['PINECONE_INDEX_NAME'] = 'ornidexpfe'

# Fill the vector database with synthetic data from our file sheet
index = 'ornidexpfe'
sheet_url = 'https://docs.google.com/spreadsheets/d/1kGwUpj97kT10rAacYi9mYVgvEk-SLixVxA8V5BfrHMo/edit?usp=sharing'
df = pd.read_csv(convert_google_sheet_url(sheet_url))

# Extract the data from the sheet
emails = []
for _, row in df.iterrows():
    emails.append(f"{row['Case title']} {row['Case description']} {row['Response']}")

emails=clean_data(emails)
#1 create a vector database if it doesn't exist
def create_vector_database(index, dimensions):
    """
    Args:
    - index (str): The name of the vector index.
    - dimensions (int): The dimensionality of the vectors.

    Returns:
    - None
    """
    pc = Pinecone(api_key=pinecone_secret_key)
    if index not in pc.list_indexes().names():
        pc.create_index(
            name = index,
            dimension = dimensions,
            metric = 'cosine',
            spec=ServerlessSpec(
                cloud="aws",
                region="us-west-2"
            )
        )
    pc.describe_index(index)

# Create a vector database if it doesn't exist
create_vector_database(index, 4096)

#2 Fills a vector database with document embeddings derived from email content
def fill_vector_database(emails):
    """
    Args:
    - emails (list): A list of email contents.

    Returns:
    - None
    """
    docs = []
    senders = ['lamine.arab@gmail.com', 'mamar.temam@gmail.com', 'celine.laglab@gmail.com']
    recievers = ['lynda.said_lhaj@gmail.com', 'alaa.adimi@gmail.com', 'rania.rezkellah@gmail.com']
    for email in emails:
        docs.append(Document(
        page_content=email,
        metadata={"Type": "Generated Email", "Sender": random.choice(senders), "Receiver": random.choice(recievers)},
))

    embeddings=CohereEmbeddings(cohere_api_key=cohere_secret_key)
    docsearch=Pinecone_Langchain.from_documents(docs, embeddings, index_name=index)

# Fill the vector database with data
fill_vector_database(emails)
