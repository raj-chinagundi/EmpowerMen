# EmpowerMen
EmpowerMen is a Question-Answering (QA) application designed to address queries related to false accusations of rape against men, enriched through a knowledge base sourced from podcasts. This project aims to provide accurate and informed responses using advanced natural language processing techniques.

## Project Flow
### Step 1: Dataset Creation for Knowledge Base
  To create the knowledge base:
  
  Provide a list of URLs to podcast videos.
  Utilize the ```DatasetCreation.ipynb```(Use google colab) notebook to generate .txt transcripts from the podcasts. These transcripts will serve as the foundational text data for the QA app.

### Step 2: Text Embedding and Vector Store Creation
```AskQuery.py```
  Use FastEmbed Embeddings from ChromaDB to convert the transcript text into embeddings.
  Store these embeddings in ChromaDB's vector store for efficient retrieval and similarity scoring.
  
### Step 3: Build Streamlit UI for QA Interface
```app.py```
Develop a Streamlit-based user interface (UI) that allows users to input questions.
Display responses generated by the QA model based on the knowledge base.

### Step 4: QA Model Implementation
```AskQuery.py```
  Implement the Ollama framework integrated with langchain to facilitate the llama3 model.
  Utilize this model to analyze and generate responses based on the enriched knowledge base of podcast transcripts.

## How to run the app:
Note: Ollama should be installed on your system.
1) Create a virtual environment ```python3 -m venv empowermen_venv``` in your preferred project directory using terminal.
2) Activate environment ```empowermen_venv\Scripts\Activate``` for windows, ```source empowermen_venv\bin\activate``` for mac on terminal.
3) Run ```pip3 install -r requirements.txt``` on terminal.
4) Run ```streamlit run app.py``` on terminal.
