# Use an official Python runtime as a parent image
FROM python:3.10

# Set the working directory
WORKDIR /rag

# Copy the requirements file
COPY requirements.txt /rag/requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY RAG.ipynb /rag
COPY llm_responder.py /rag
COPY llm_app.py /rag
COPY init.sh /rag
COPY blogtext_small.csv /rag

RUN apt-get update && apt-get install -y supervisor

COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Expose the port Jupyter uses
EXPOSE 8080
# Expose the port Streamlit uses
EXPOSE 8501

ENTRYPOINT ["/rag/init.sh"]
