import re
from typing import List, Dict, Any
from data_handling import *
from langchain.schema import Document
from configparser import ConfigParser
from langchain_community.embeddings import CohereEmbeddings
from langchain_community.vectorstores import  Pinecone as Pinecone_Langchain


config = ConfigParser()
config.read("config.ini")
cohere_secret_key = config.get('Cohere', 'secret_key')
pinecone_secret_key = config.get('Pinecone', 'secret_key')
index = 'ornidexpfe'

#1: Extract the document information
def extract_document_info(responses):
    """
    Extracts document information from a list of responses.

    Args:
    - responses (list): A list of response dictionaries containing document metadata.

    Returns:
    - list: A list of document texts from responses with a score greater than 0.2.
    """
    return [response["metadata"]["text"] for response in responses['matches'] if response["score"] > 0.2]

#3: A function to transform the Google sheet url to a downloadable CSV url
def convert_google_sheet_url(url):
    """

    Args:
    - url (str): The Google Sheets URL to convert.

    Returns:
    - str: The converted URL pointing to a downloadable CSV.
    """
    
    # Regular expression pattern to match Google Sheets URL
    pattern = r'https://docs\.google\.com/spreadsheets/d/([a-zA-Z0-9-_]+)(/edit#gid=(\d+)|/edit.*)?'
    # Replacement function to convert the URL
    replacement = lambda m: f'https://docs.google.com/spreadsheets/d/{m.group(1)}/export?' + (f'gid={m.group(3)}&' if m.group(3) else '') + 'format=csv'
    new_url = re.sub(pattern, replacement, url)
    return new_url

# Create received data file
# Vector database augmentation with recieved databatches
def process_and_ingest(batch: Dict[str, Any]):
    
    """
    Process and ingest a batch of data recieved from the CRM plateform
    into the vector database.

    Parameters:
    - batch (Dict[str, Any]): The batch of data containing Salesforce
    object name and data entries.

    The batch should have the following structure:
    {
        "sf_object_api_name": "Case",
        "Data": [
            {"case_name": "...", "case_title": "..."}
        ]
    }
    """
    text = []
    types = []
    docs = []
    # Extract the Salesforce object name from the batch
    sf_object_name = batch.get('sf_object_api_name')
    # Extract the data entries from the batch
    data_entries = batch.get('Data', [])
    for data_entry in data_entries:
        tex = ''
        for key, value in data_entry.items():
            # Concatenate all values in the data entry into a single string
            tex += str(value) + " " 

        text.append(tex.strip()) 
        types.append(sf_object_name)
    
    # Clean the text data using a custom clean_data function
    texts = clean_data(text)
    # Create Document objects for each cleaned text with its corresponding metadata
    for i in range(len(texts)):
        docs.append(Document(
        page_content = texts[i],
        metadata = {"Type": types[i]},
        ))
    embeddings = CohereEmbeddings(cohere_api_key=cohere_secret_key)
    
    # Ingest the documents into the Pinecone vector database 
    docsearch = Pinecone_Langchain.from_documents(docs, embeddings, index_name=index)
