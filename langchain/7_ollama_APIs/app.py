## Load all the keys 
import os 
from dotenv import load_dotenv
from langchain_community.llms import Ollama
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
print("Welcome to streamlit application")

# For Langsmith Tracking
os.environ['LANGCHAIN_API_KEY']=os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_TRACING_V2']="true"
os.environ['LANGCHAIN_PROJECT']=os.getenv("LANGCHAIN_PROJECT")

# Design your Prompt Template
prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please respond to the question asked"),
        ("user","Question:{question}")
    ]

)

# Design streamlit application
# Install Ollama on Ubuntu WSL using "curl -fsSL https://ollama.com/install.sh | sh"
# Run ollama with "ollama serve" and leave the termimal open
# Pull ollama with "ollama pull gemma:2b"
# Run "streamlit runa app.py" in another WSL terminal 
# Monitor these API calls with LangSmith at smith.langchain.com/profile/settings/yourproject

st.title("Langchain Application with Gemma Model")
input_text=st.text_input("Ask anything...")

# Ollama Llama2 model (make sure this model pulled on local machine)
llm=Ollama(model="gemma:2b")
output_parser=StrOutputParser()
chain=prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({"question":input_text}))

