# Document-Retrieval-System
Document Retrieval System using RAG and LangChain.

## Overview
This project is a Python application designed to perform document retrieval using Retrieval-Augmented Generation (RAG) and LangChain. It uses kaggle blog collection as dataset from here: [blog collection](https://www.kaggle.com/datasets/rtatman/blog-authorship-corpus/data). The data is processed and chunked, and embeddings are created for these chunks. The chunks are then stored in a Chroma DB vector database, which is managed within a Docker container. When a user asks a question, the application embeds the query, retrieves relevant documents from the database and passes both the question and the documents to a language model (LLM) for a response. The system also supports chat history, enabling connected questions for the LLM.

## Files
<ol>
  <li>blogtext_small.csv: Dataset used for RAG, filtered from the original dataset on Kaggle.</li>
  <li>chroma_db: folder with already prepared data (chunked and ready for chroma vector database).</li>
  <li>init.sh: script the runs as entrypoint to app docker container</li>
  <li>add_data_to_collection.py: Script to create data collection from the dataset for loading into Chroma DB.</li>
  <li>llm_app.py: Streamlit application for testing model responses to dataset questions.</li>
  <li>llm_responder.py: Code for generating LLM responses.</li>
  <li>requirements.txt: List of necessary Python packages.</li>
  <li>RAG.ipynb: Jupyter notebook illustrating the RAG process, useful for reference.</li>
  <li>Dockerfile: Initializes the Docker container with Python code, Jupyter server, and Streamlit server.</li>
  <li>docker-compose.yml: Defines and manages Chroma DB and application Docker containers.</li>
  <li>supervisord.conf: Configuration to start both Jupyter and Streamlit servers in the Docker container.</li>
  <li>populate_vector_db.py: Script to download data and store it in the collection, useful for on-demand collection creation inside Docker.</li>
</ol>

## Setup and Usage
There are two ways to setup the project:
Directly download the image from Docker Hub:
<ol>
  <li>
    download and extract zip from: https://drive.google.com/file/d/1Uj2g2FaoYMYJafe5xdkV71aHFYsFpKPB/view?usp=sharing
  </li>
  <li> run "docker-compose up"</li>
  <li>open http://localhost:8501 for the streamlit app to test the model response (to run the model you need to set Hugging Face access token you can generate on their site)</li>
  <li>open http://localhost:8080 for the jupyter server with the RAG storybook</li>
</ol>
Build the image from the repo:
<ol>
  <li>Clone the repository. <br>
      NOTE: add flag "-c core.autocrlf=false" to your git clone command to prevent auto conversion to crlf file endings. For example: git clone -c core.autocrlf=false https://github.com/MladenDzida/Document-Retrieval-System</li>
  <li>Build and run Docker containers: docker-compose up</li>
  <li>open http://localhost:8501 for the streamlit app to test the model response (to run the model you need to set Hugging Face access token you can generate on their site)</li>
  <li>open http://localhost:8080 for the jupyter server with the RAG storybook</li>
</ol>

