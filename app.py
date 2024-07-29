import streamlit as st
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import os
from openai import AuthenticationError
from operator import itemgetter
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from  langchain.memory import ConversationBufferWindowMemory
import time
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    )
from langchain_openai import OpenAI


st.header("Chat with PDF 💬")
subheader = st.subheader('Enter an API key in the sidebar to chat with your pdf.',divider=True)
if "OpenAPIKey" not in st.session_state:
        st.session_state.OpenAPIKey = None
with st.sidebar:
    st.title('🤗💬 LLM Chat App')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - https://streamlit.io/
    - https://python.langchain.com/
    - https://platform.openai.com/docs/models LLM model
    ''')
    openai_api_key = st.text_input("Enter OpenAI API Key", type="password")
    if openai_api_key:
        st.session_state.OpenAPIKey = openai_api_key
    add_vertical_space(5)
        

def extract_text_from_pdf(pdf):
    """
    Extracts text from a PDF file.

    Args:
        pdf: The PDF file to extract text from (can be a file path or file-like object).

    Returns:
        str: Extracted text from the PDF.
    """
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def process_text_and_load_vector_store(text):
    """
    Splits the given text into chunks, loads or creates a vector store, and returns it.

    Args:
        text (str): The text to be split and processed.

    Returns:
        VectorStore: The vector store loaded from a file or created from the text chunks.
    """
                
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
        )
    chunks = text_splitter.split_text(text=text)
    store_name = pdf.name[:-4]
    if os.path.exists(f"{store_name}.pkl"):
        with open(f"{store_name}.pkl", "rb") as f:
            VectorStore = pickle.load(f)
    else:
        embeddings=HuggingFaceEmbeddings()
        VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
        with open(f"{store_name}.pkl", "wb") as f:
            pickle.dump(VectorStore, f)
    return VectorStore

def main(query):
    """
    Main function to handle PDF text extraction, vector store processing, and user query handling.
    
    - Extracts text from a PDF file.
    - Processes the text and loads or creates a vector store.
    - Accepts user input for queries about the PDF.
    - Uses an LLM to generate a response based on the similarity search results.
    """

    try:
        text = extract_text_from_pdf(pdf)
        VectorStore = process_text_and_load_vector_store(text)
        docs = VectorStore.similarity_search(query=query, k=3)
        chain = (
            RunnablePassthrough.assign(
                history=RunnableLambda(
                    st.session_state.memory.load_memory_variables) | itemgetter("history"),
            )
            | prompt_templates
            | llm
        )
        output = ""
        for chunk in chain.stream({"question":query, "input_documents": docs}):
            output += chunk.content
            yield chunk.content
            time.sleep(0.05)

        st.session_state.memory.save_context({"inputs": query}, {"output": output})

    except AuthenticationError:
        st.warning(
            body='AuthenticationError : Please provide correct api key 🔑' ,icon='🤖')
    except Exception as e:
        st.warning(f"An error occure while processing the query: {e}")


if __name__ == '__main__':
    if st.session_state.OpenAPIKey:
        subheader.empty()

        pdf = st.file_uploader("Upload your PDF", type='pdf')
        llm = OpenAI(api_key=st.session_state.OpenAPIKey,temperature=0,verbose=True)
        if "memory" not in st.session_state:
            st.session_state.memory = ConversationBufferWindowMemory(
                llm=llm,
                memory_key="history",
                return_messages=True,
                k=10
            )
        
        prompt_templates = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(
                "{input_documents}"),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{question}")], input_variables=["input_documents"])
        if pdf:
            if "messages" not in st.session_state:
                    st.session_state.messages = []

            for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

            if prompt := st.chat_input("Ask Query?", key='QueryKeyForTextInput'):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                with st.chat_message("assistant"):  
                    with st.spinner("Thinking..."):                      
                        full_response=st.write_stream(main(prompt))
                        st.session_state.messages.append({"role": "assistant", "content": full_response})