import streamlit as st
import pandas as pd
from llm_responder import generate_llm_reponse
import os
from langchain_core.messages import HumanMessage

# Load the data
df = pd.read_csv('blogtext_small.csv', delimiter=',')
df = df.filter(['text'])
number_of_rows = len(df)

st.title("Chat with LLM")

# Input for Hugging Face access token
HUGGINGFACEHUB_API_TOKEN = st.text_input("Enter your Hugging Face access token:", type="password")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

# store chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

st.write("Preview data:")
st.dataframe(df, use_container_width=True)

# Input field for question
question = st.text_input("Enter your question for the LLM:")

llm, history = st.columns([1, 1])
# Button to get LLM response
with llm:
    if st.button("Get LLM Response"):
        if question:
            response = generate_llm_reponse(question, st.session_state.chat_history)
            st.session_state.chat_history.extend([HumanMessage(content=question), response["answer"]])
            st.write("LLM Response:")
            st.write(response["answer"])
        else:
            st.write("Please enter a question:")
with history:
    if st.button("Clear chat history"):
        st.session_state.chat_history = []