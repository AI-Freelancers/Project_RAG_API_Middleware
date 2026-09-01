from fastapi import FastAPI, HTTPException, BackgroundTasks
from utils import *
from pydantic import BaseModel
from base_models import *
from specialized_models import *
from contextualized_models import *
from finetuned_models import *

'''The API '''
# Create the FastAPI app
app = FastAPI()
# Request Model
class RequestModel(BaseModel):
    model: str
    type: str
    message: str = None
    doc_type: str
    sender: str
    receiver: str

# Define the root route
@app.get("/")
async def root():
    """
    Welcome message for the root endpoint.
    """
    return {"message": "Welcome to the Email generation API!"}

# Model Selection Endpoint
@app.post("/llm")
async def model_selection(request: RequestModel):
    """
    Selects the appropriate model and type based on the input parameters and returns the response.

    Args:
    - request (RequestModel): An instance of the RequestModel class containing model, type, and message.

    Returns:
    - dict: A dictionary containing the response.
    """
    model = request.model.lower()
    type = request.type.lower()
    message = request.message
    receiver = request.receiver
    sender = request.sender
    doc_type = request.doc_type

    metadata = {}
    if receiver != '':
        metadata['Receiver'] = receiver
    if sender != '':
        metadata['Sender'] = sender
    if doc_type != '':
        metadata['doc_type'] = doc_type

    # Cohere API
    if model == 'cohere':
        if type == 'base':
            response = cohere_base_call(message)
        elif type == 'contextualized':
            response = cohere_contextualized_call(message)
        elif type == 'finetuned':
            response = cohere_finetuned_call(message)
        elif type == 'specialized':
            response = cohere_specialized_call(message, metadata)
        else:
            response = "This model type is not supported yet."
        return {"response": response}
    # OpenAI API
    elif model == 'openai':
        if type == 'base':
            response = openai_base_call(message)
        elif type == 'contextualized':
            response = openai_contextualized_call(message)
        elif type == 'finetuned':
            response = openai_finetuned_call(message)
        elif type == 'specialized':
            response = openai_specialized_call(message, metadata)
        else:
            response = "This model type is not supported yet."
        return {"response": response}
    # AI21 API
    elif model == 'ai21':
        if type == 'base':
            response = ai21_base_call(message)
        elif type == 'contextualized':
            response = ai21_contextualized_call(message)
        elif type == 'specialized':
            response = ai21_specialized_call(message, metadata)
        else:
            response = "This model type is not supported yet."
        return {"response": response}
    elif model == 'palm':
        if type == 'base':
            response = palm_base_call(message)
        elif type == 'contextualized':
            response = palm_contextualized_call(message)
        elif type == 'specialized':
            response = palm_specialized_call(message, metadata)
        else:
            response = "This model type is not supported yet."
        return {"response": response}
    else:
        return {"response": "This model is not supported yet."}

# Define the endpoint for batch data gathering

@app.post("/ingest")
async def ingest_data(data_batch: List[Dict[str, Any]] , background_tasks: BackgroundTasks):
    try:
        # Start the background task
        background_tasks.add_task(process_and_ingest, data_batch)
        
        return {"status": "success", "message": "Data ingestion started."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
