# Importing necessary libraries
import json
import os
import sys
import boto3
import streamlit as st

# We will be using Amazon Titan embeddings model for generating embeddings
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

# Importing libraries for Data Ingestion
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

# For converting to Vector Embeddings and Vector Stores
from langchain.vectorstores import FAISS

# LLM Models
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Bedrock client initialization
bedrock = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)

# Data Ingestion
def data_ingestion():
    loader = PyPDFDirectoryLoader('data/')
    documents = loader.load()

    # Using Character text splitter to split the documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    docs = text_splitter.split_documents(documents)
    return docs

# Vector Embeddings & Vector Store Creation
def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(documents=docs, embedding=bedrock_embeddings)
    vectorstore_faiss.save_local("faiss_index")

from langchain_community.chat_models import BedrockChat

# Claude LLM
def get_claude_llm():
    # llm = Bedrock(model_id="anthropic.claude-3-haiku-20240307-v1:0", 
    #               client=bedrock,
    #               model_kwargs = {"max_tokens_to_sample": 512})
    llm = BedrockChat(
        model_id="anthropic.claude-3-haiku-20240307-v1:0",
        client=bedrock,
        model_kwargs={"max_tokens": 512}   # note: `max_tokens` not `max_tokens_to_sample`
    )
    return llm

# Amazon Titan LLM
def get_titan_llm():
    llm = Bedrock(model_id="amazon.titan-text-lite-v1", 
                  client=bedrock,
                  model_kwargs = {"maxTokenCount": 512})
    return llm

# Prompt Template
prompt_template = """
Human: Use the following pieces of context to provide a detailed and 
concise answer to the question at the end. Your answer should be 
well-explained and at least 250 words long. 

If you don't know the answer, simply respond with "I don't know" 
— do not attempt to make up an answer.

<context>
{context}
</context>

Question: {question}

Assistant:
"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

def get_response_llm(llm, vectorstore_faiss, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    answer = qa({"query": query})
    return answer['result']

# Streamlit App
def main():
    st.set_page_config("Chat PDF")
    
    st.header("Chat with PDF using AWS Bedrock 💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    with st.sidebar:
        st.title("Update Or Create Vector Store:")
        
        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("Done")

    if st.button("Claude Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", 
                                           bedrock_embeddings, 
                                           allow_dangerous_deserialization=True)
            llm=get_claude_llm()
            
            st.write(get_response_llm(llm,faiss_index,user_question))
            st.success("Done")

    if st.button("Titan Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", 
                                           bedrock_embeddings,
                                           allow_dangerous_deserialization=True)
            llm=get_titan_llm()

            st.write(get_response_llm(llm,faiss_index,user_question))
            st.success("Done")

if __name__ == "__main__":
    main()