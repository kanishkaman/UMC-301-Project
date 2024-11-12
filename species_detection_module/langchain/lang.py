from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Loas the groq api key from env folder
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")

## Prompt Template
prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are an ecologist and wildlife expert. The output should tell about the species, its natural habitat, its conservation status and other interesting facts about it. If the input is no species detected prompt the user about it. "),
        ("user","Question:{question}")
    ]
)
## Streamlit framework

output_parser=StrOutputParser()
groq_chat = ChatGroq(temperature=0, groq_api_key = groq_api_key , model_name="llama3-8b-8192")
groq_chain = prompt | groq_chat | output_parser
