# Customize LLMs for Specific Business Processes

LLMs Personalization for specific Business processes (Use-case: Emails Generation)

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)

## Introduction

Businesses today face a growing barrage of customer inquiries, internal communications, and workflow demands. Managing these effectively requires efficient, personalized, and accurate communication, often exceeding human capabilities.
The goal of this project is To develop and implement a customizable LLM-powered personalization framework that can be seamlessly integrated within any CRM platform, empowering businesses to overcome the limitations of traditional communication and knowledge management methods.
### Specific Focus
Within the broader scope, the initial implementation and use case will focus on personalized email generation within Salesforce as a representative CRM platform.
### Future Scalability
The framework is designed with modularity and flexibility in mind, allowing for future adaptation and integration with other CRM platforms and extending its functionality to additional use cases beyond email generation.

## Features
Four models versions are used in this project:
- A base version: the output will only depend on the model used and the original query.
- A contextualized version: the output will depend on the user's query, the prompts guiding the model, and the model's nature.
- A finetuned version: the models are finetuned for the specific usecase using a synthetic dataset.
- A specialized version: which is the core of our solution, this is done using ** Retrieval Augmented Generation**.

Our system is encapsulated in an API with three two main endpoints: a model selection endpoint, and a data gathering (scheduled batches) endpoint.
This API leverages 4 different Large Language Models from the following organizations: Openai, Cohere, AI21, and Google PaLM.

 ## Installation
- First of all, you need to create a virtual environment, using this command:
 ```bash
python -m venv llms-api
```
- Then activate it using this command:
```bash
./llms-api/Scripts/activate
```
- Move to the app directory and install all the dependencies in 'requirements.txt':
```bash
pip install -r requirements.txt --upgrade
```
- Run the following command to ensure the privacy layer works correctly:
```bash
python -m spacy download en_core_web_lg
```

## Usage
We make use of uvicorn in order to run the server, so before making calls we need to run the server:
```bash
uvicorn main:app --reload 
```



