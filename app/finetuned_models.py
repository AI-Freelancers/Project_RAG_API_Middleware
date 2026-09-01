from configparser import ConfigParser
import cohere
import openai


# Read the secret keys from the configuration file
config = ConfigParser()
config.read("config.ini")
cohere_secret_key = config.get('Cohere', 'secret_key')
openai_secret_key = config.get('OpenAI', 'secret_key')

'''Functions'''
#1 Makes a finetuned call to the Cohere API.
def cohere_finetuned_call(message):
    """
    Args:
    - message (str): The input prompt for generation.

    Returns:
    - str: The generated text response.
    """
    co = cohere.Client(cohere_secret_key)
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
        model='0befb7b9-c512-4d37-b454-268a6a73eb65-ft',
        stop_sequences=[],
        return_likelihoods='NONE'
    )
    return response.generations[0].text

#2 Makes a finetuned call to the OpenAI API.
def openai_finetuned_call(message):
    """
    Args:
    - message (str): The input prompt for generation.

    Returns:
    - str: The generated text response.
    """
    openai.api_key = openai_secret_key
    response = openai.chat.completions.create(
        model='ft:gpt-3.5-turbo-0613:personal::8EvhfHex',
        messages=[
            {"role":"user", "content": f"""You are a chatbot that writes personalized emails containing answers to cases that we receive on our Salesforce CRM. 
                                            Write an email with a subject and a body, and sign it as 'The Ornidex Team'. The email should address the case owner and incorporate the necessary details from the case description and agent input to respond to the following query: {message}
                                            Please generate the email response in the following format:
                                
                                            Subject: [Title]
                                            Dear [Account name],
                                            
                                            [Personalized email body addressing the case description and agent input.]
                                            
                                            Best regards,
                                            The Ornidex Team
                                            """}
        ]
    )
    return response.choices[0].message.content

def ai21_finetuned_call(message):
    pass

def palm_finetuned_call(message):
    pass
