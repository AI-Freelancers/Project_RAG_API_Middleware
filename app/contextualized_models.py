from configparser import ConfigParser
import cohere
import openai
from ai21 import AI21Client
from ai21.models import RoleType
from ai21.models import ChatMessage
import google.generativeai as palm


# Read the secret keys from the configuration file
config = ConfigParser()
config.read("config.ini")
cohere_secret_key = config.get('Cohere', 'secret_key')
openai_secret_key = config.get('OpenAI', 'secret_key')
ai21_secret_key = config.get('AI21', 'secret_key')
palm_secret_key = config.get('PaLM', 'secret_key')

'''Functions'''
#1 Makes a contextualized call to the Cohere API.
def cohere_contextualized_call(message):
    """
    Args:
    - message (str): The query to be included in the email.

    Returns:
    - str: The generated text response.
    """
    co = cohere.Client(cohere_secret_key)
    response = co.generate(
        prompt= f"""You are a chatbot that writes personalized emails containing answers to cases that we receive on our Salesforce CRM. 
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
    return response.generations[0].text

#2 Makes a contextualized call to the OpenAI API.
def openai_contextualized_call(message):
    """
    Args:
    - message (str): The query to be included in the email.

    Returns:
    - str: The generated text response.
    """
    openai.api_key = openai_secret_key
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
    return response.choices[0].message.content

#3 Makes a contextualized call to the AI21 API.
def ai21_contextualized_call(message):
    """
    Args:
    - message (str): The query to be included in the email.

    Returns:
    - str: The generated text response.
    """
    client = AI21Client(api_key= ai21_secret_key)
    system = "You are a chatbot that writes personalized emails containing answers to cases that we receive on our Salesforce CRM. "
    messages = [
        ChatMessage(text=f"""Write an email with a subject and a body, and sign it as 'The Ornidex Team'. The email should address the case owner and incorporate the necessary details from the case description and agent input to respond to the following query: {message}
                    Please generate the email response in the following format:
        
                    Subject: [Title]
                    Dear [Account name],
                    
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
    return response.outputs[0].text

#4 Makes a contextualized call to the PaLM API.
def palm_contextualized_call(message):
    """
    Args:
    - message (str): The query to be included in the email.

    Returns:
    - str: The generated text response.
    """
    palm.configure(api_key=palm_secret_key)
    completion = palm.generate_text(
        model="models/text-bison-001",
        prompt= f"""Write an email with a subject and a body, and sign it as 'The Ornidex Team'. The email should address the case owner and incorporate the necessary details from the case description and agent input to respond to the following query: {message}
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
    return completion.result
