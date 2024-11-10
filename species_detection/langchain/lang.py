from langchain.llms import OpenAI
from langchain.agents import initialize_agent, Tool
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import WikipediaLoader
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import AgentExecutor

# Step 1: Load Wikipedia data (Wikipedia Loader)
def load_wikipedia_data(species_name):
    loader = WikipediaLoader(query=species_name, num_results=5)  # Load 5 results from Wikipedia
    documents = loader.load()
    return documents

# Step 2: Initialize embeddings and retriever
def initialize_retriever(documents):
    # Create embeddings and vectorstore (FAISS for semantic search)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})  # Retrieve top 3 documents
    return retriever

# Step 3: Define a prompt template for the LLM
species_prompt = """
You are a knowledgeable species expert. The following information is provided to you about a species. Based on the context, provide a detailed summary about the species, including its habitat, conservation status, and other useful facts.

Species Name: {species_name}
Context: {context}

Please provide the summarized information.
"""

prompt = PromptTemplate(input_variables=["species_name", "context"], template=species_prompt)

# Step 4: Initialize the LLM and the LLM Chain
# Import the api key from .env file

import os
import getpass

# Set the Google API key

os.environ["GOOGLE_API_KEY"] = getpass.getpass()

from langchain_google_vertexai import ChatVertexAI

llm = ChatVertexAI(model="gemini-1.5-flash")
llm_chain = LLMChain(prompt=prompt, llm=llm)

# Step 5: Define the RetrievalQA Chain
def create_retrieval_qa_chain(retriever, llm_chain):
    # This chain will first retrieve documents from the Wikipedia data and then pass them to the LLM for summarization
    qa_chain = RetrievalQA(retriever=retriever, combine_docs_chain=llm_chain)
    return qa_chain

# Step 6: Integrating the pipeline
def generate_species_info(species_name):
    # Load data related to the species from Wikipedia
    documents = load_wikipedia_data(species_name)

    # Initialize the retriever with the documents
    retriever = initialize_retriever(documents)

    # Create a retrieval-augmented generation (RAG) pipeline using the QA chain
    qa_chain = create_retrieval_qa_chain(retriever, llm_chain)

    # Query the pipeline with the species name
    result = qa_chain.run(species_name)
    
    return result

# Step 7: Testing the pipeline
if __name__ == "__main__":
    species_name = "Panda"
    species_info = generate_species_info(species_name)
    print("Species Information:", species_info)
