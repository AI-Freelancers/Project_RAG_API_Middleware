from configparser import ConfigParser
import cohere
import openai
import google.generativeai as palm
import os
from langchain_community.embeddings import CohereEmbeddings
from pinecone import Pinecone
from langchain_community.vectorstores import Pinecone as Pinecone_Langchain
from data_handling import *
from utils import *
from ai21 import AI21Client
from ai21.models import RoleType, Penalty
from ai21.models import ChatMessage

# Read the secret keys from the configuration file
config = ConfigParser()
config.read("config.ini")
cohere_secret_key = config.get('Cohere', 'secret_key')
openai_secret_key = config.get('OpenAI', 'secret_key')
ai21_secret_key = config.get('AI21', 'secret_key')
palm_secret_key = config.get('PaLM', 'secret_key')
pinecone_secret_key = config.get('Pinecone', 'secret_key')
os.environ['PINECONE_API_KEY'] = pinecone_secret_key
os.environ['PINECONE_INDEX_NAME'] = 'ornidexpfe'
os.environ['COHERE_API_KEY'] = cohere_secret_key

""" Retrieval Augmented Generation Functions"""
#1 Makes a specialized call to the Cohere API.
def cohere_specialized_call(message, metadata):
    """
    Args:
    - message (str): The query of the user.

    Returns:
    - str: The generated text response using the anonymized retrieved 
    context, to respond to the user query
    """
    co = cohere.Client(cohere_secret_key)
    embeddings = CohereEmbeddings(cohere_api_key=cohere_secret_key)
    query = embeddings.embed_query(message)
    pc = Pinecone(api_key=pinecone_secret_key)
    index = pc.Index('ornidexpfe')
    results = index.query(
        vector=query,
        top_k=2,
        include_metadata=True,
        filter= metadata
    )
    print(results)
    results = extract_document_info(results)
    #results.insert(0,message)
    #an, text=privacy_layer(results)
    if len(results) == 0:
        response = co.generate(
            prompt=f"""You are a chatbot that writes personalized emails containing answers to cases that we receive on our Salesforce CRM. 
                    Write an email with a subject and a body, and sign it as 'The Ornidex Team'. The email should address the case owner and incorporate the necessary details from the case description and agent input to respond to the following query: {message}
                    Please generate the email response in the following format:
        
                    Subject: [Title]
                    Dear [Account name],
                    
                    [Personalized email body addressing the case description and agent input.]
                    
                    Best regards,
                    The Ornidex Team
                    """,
            model='command-xlarge-nightly',
            max_tokens=800,
            temperature=0.2,
            stop_sequences=[],
            return_likelihoods='NONE'
        )
    else:
        response = co.generate(
            prompt=f"""You are a chatbot that writes a single personalized email response to a Salesforce case. The email should be addressed to the case owner, incorporate the necessary details from the case description and agent input, and rely on the relevant documents as needed. The email should be signed by The Ornidex Team.
                    Here is the context from Salesforce: {message}
                    Here are the relevant documents: {results}
                    
                    Please generate one email response in the following format:
                    
                    Subject: [Title]
                    
                    Dear [Account name],
                    
                    [Personalized email body addressing the case description and agent input, referencing relevant documents as needed.]
                    
                    Best regards,
                    The Ornidex Team
                    """,
            model='command-xlarge-nightly',
            max_tokens=800,
            temperature=0.2,
            stop_sequences=[],
            return_likelihoods='NONE'
        )
    #deanonymize_data(an, response.generations[0].text)  
    return response.generations[0].text

#2 Makes a specialized call to the OpenAI API.
def openai_specialized_call(message, metadata):
    """
    Args:
    - message (str): The query of the user.

    Returns:
    - str: The generated text response using the anonymized retrieved 
    context, to respond to the user query
    """
    openai.api_key = openai_secret_key
    embeddings = CohereEmbeddings(cohere_api_key=cohere_secret_key)
    query = embeddings.embed_query(message)
    pc = Pinecone(api_key=pinecone_secret_key)
    index = pc.Index('ornidexpfe')
    results = index.query(
        vector=query,
        top_k=2,
        include_metadata=True,
        filter= metadata
    )
    results = extract_document_info(results)
    #results.insert(0,message)
    #an, text = privacy_layer(results)
    if len(results) == 0:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": f"""You are a chatbot that writes personalized emails containing answers to cases that we receive on our Salesforce CRM. 
                    Write an email with a subject and a body, and sign it as 'The Ornidex Team'. The email should address the case owner and incorporate the necessary details from the case description and agent input to respond to the following query: {message}
                    Please generate the email response in the following format:
        
                    Subject: [Title]
                    Dear [Account name],
                    
                    [Personalized email body addressing the case description and agent input.]
                    
                    Best regards,
                    The Ornidex Team
                    """}],
            max_tokens=800,
            temperature=0.2,
        )
    else:
        response = openai.chat.completions.create(
            model='gpt-3.5-turbo',
            messages=[
                {"role":"user", "content": f"""You are a chatbot that writes a single personalized email response to a Salesforce case. The email should be addressed to the case owner, incorporate the necessary details from the case description and agent input, and rely on the relevant documents as needed. The email should be signed by The Ornidex Team.
                    Here is the context from Salesforce: {message}
                    Here are the relevant documents: {results}
                    
                    Please generate one email response in the following format:
                    
                    Subject: [Title]
                    
                    Dear [Account name],
                    
                    [Personalized email body addressing the case description and agent input, referencing relevant documents as needed.]
                    
                    Best regards,
                    The Ornidex Team
                    """}
            ],
            max_tokens=800,
            temperature=0.2,
        )
    # deanonymize_data(an, response.choices[0].text)   
    return response.choices[0].message.content

#3 Makes a specialized call to the AI21 API.
def ai21_specialized_call(message, metadata):
    """
    Args:
    - message (str): The query of the user.

    Returns:
    - str: The generated text response using the anonymized retrieved 
    context, to respond to the user query
    """
    client = AI21Client(api_key= ai21_secret_key)
    system = "You are a chatbot that writes personalized emails containing answers to cases that we receive on our Salesforce CRM."
    embeddings = CohereEmbeddings(cohere_api_key=cohere_secret_key)
    query = embeddings.embed_query(message)
    pc = Pinecone(api_key=pinecone_secret_key)
    index = pc.Index('ornidexpfe')
    results = index.query(
        vector=query,
        top_k=2,
        include_metadata=True,
        filter= metadata
    )
    results = extract_document_info(results)
    #results.insert(0,message)
    #an, text = privacy_layer(results)
    if len(results) == 0:
        messages = [
            ChatMessage(text=f"""
                    Write an email with a subject and a body, and sign it as 'The Ornidex Team'. The email should address the case owner and incorporate the necessary details from the case description and agent input to respond to the following query: {message}
                    Please generate the email response in the following format:
        
                    Subject: [Title]
                    Dear [Case Owner],
                    
                    [Personalized email body addressing the case description and agent input.]
                    
                    Best regards,
                    The Ornidex Team
                    """, role=RoleType.USER),
        ]
        response = client.chat.create(
            system=system,
            messages=messages,
            model="j2-ultra",
        )
    else:
        messages = [
            ChatMessage(text=f"""You are a chatbot that writes a single personalized email response to a Salesforce case. The email should be addressed to the case owner, incorporate the necessary details from the case description and agent input, and rely on the relevant documents as needed. The email should be signed by The Ornidex Team.
                        Here is the context from Salesforce: {message}
                        Here are the relevant documents: {results}
                        
                        Please generate one email response in the following format:
                        
                        Subject: [Title]
                        
                        Dear [Account name],
                        
                        [Personalized email body addressing the case description and agent input, referencing relevant documents as needed.]
                        
                        Best regards,
                        The Ornidex Team
                    """, role=RoleType.USER),
        ]
        response = client.chat.create(
            system=system,
            messages=messages,
            model="j2-ultra",
        )
    # deanonymize_data(an, response.outputs[0].text)  
    return  response.outputs[0].text

#4 Makes a specialized call to the PaLM API.
def palm_specialized_call(message, metadata):
    """
    Args:
    - message (str): The query of the user.

    Returns:
    - str: The generated text response using the anonymized retrieved 
    context, to respond to the user query
    """
    palm.configure(api_key=palm_secret_key)
    embeddings = CohereEmbeddings(cohere_api_key=cohere_secret_key)
    query = embeddings.embed_query(message)
    pc = Pinecone(api_key=pinecone_secret_key)
    index = pc.Index('ornidexpfe')
    results = index.query(
        vector=query,
        top_k=2,
        include_metadata=True,
        filter= metadata
    )
    results = extract_document_info(results)
    #results.insert(0,message)
    #an, text=privacy_layer(results)
    if len(results) == 0:
        completion = palm.generate_text(
            model="models/text-bison-001",
            prompt= f"""You are a chatbot that writes personalized emails containing answers to cases that we receive on our Salesforce CRM. 
                    Write an email with a subject and a body, and sign it as 'The Ornidex Team'. The email should address the case owner and incorporate the necessary details from the case description and agent input to respond to the following query: {message}
                    Please generate the email response in the following format:
        
                    Subject: [Title]
                    Dear [Account name],
                    
                    [Personalized email body addressing the case description and agent input.]
                    
                    Best regards,
                    The Ornidex Team
                    """,
            temperature=0.25,
            max_output_tokens=800,
        )
    else:
        completion = palm.generate_text(
            model="models/text-bison-001",
            prompt= f"""You are a chatbot that writes a single personalized email response to a Salesforce case. The email should be addressed to the case owner, incorporate the necessary details from the case description and agent input, and rely on the relevant documents as needed. The email should be signed by The Ornidex Team.
                    Here is the context from Salesforce: {message}
                    Here are the relevant documents: {results}
                    
                    Please generate one email response in the following format:
                    
                    Subject: [Title]
                    
                    Dear [Account name],
                    
                    [Personalized email body addressing the case description and agent input, referencing relevant documents as needed.]
                    
                    Best regards,
                    The Ornidex Team
                    """,
            temperature=0.25,
            max_output_tokens=800,
        )
    # deanonymize_data(an,completion.result)
    return completion.result
