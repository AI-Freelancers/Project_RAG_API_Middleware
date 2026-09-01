import re
from langchain_experimental.data_anonymizer import PresidioReversibleAnonymizer

'''
This line is to run before launching the system: 
!python -m spacy download en_core_web_lg
'''
#1 Cleans the data by removing leading spaces, 
# HTML tags, special characters, and multiple spaces.
def clean_data(data):
    """
    Args:
    - data (list): A list of documents to be cleaned.

    Returns:
    - list: A list of cleaned documents.
    """
    clean_data = []
    for doc in data:
        # Remove leading spaces
        doc = doc.strip()
        # Remove HTML tags
        doc = re.sub(r'<[^>]*>', '', doc)
        # Remove special characters
        doc = re.sub(r'[^a-zA-Z0-9\s]', '', doc)
        # Remove multiple spaces
        doc = re.sub(r'\s+', ' ', doc)
        clean_data.append(doc)
    return clean_data

#2 Applies reversible anonymization to sensitive information (PII) in the given context.
def privacy_layer(context):
    """
    Args:
    - context (list): A list of documents containing sensitive information.

    Returns:
    - tuple: A tuple containing the anonymizer object and a list of anonymized documents.
    """
    
    an= PresidioReversibleAnonymizer(
    analyzed_fields=["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "CREDIT_CARD", "IP_ADDRESS", "MEDICAL_LICENSE", "US_PASSPORT", "US_SSN"],
    faker_seed=None,
    # Faker seed is used here to make sure the same fake data is generated for the test purposes
    # In production, it is recommended to remove the faker_seed parameter (it will default to None)
    )
    anonymized_data=[]
    for doc in context:
        anonymized_data.append(an.anonymize(text=doc))     
    return  an,anonymized_data

#3 Deanonymizes the text using the provided anonymizer
def deanonymize_data(an, text):
    """
    Args:
    - an: The anonymizer object used for anonymization.
    - text (str): The anonymized text to be deanonymized.
    Returns:
    - str: The deanonymized text.
    """
    return an.deanonymize(text)
