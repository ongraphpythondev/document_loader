import streamlit as st
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import HuggingFaceEmbeddings
import tiktoken
from langchain.callbacks import get_openai_callback
import os
from openai import AuthenticationError 

st.header("Chat with PDF 💬")
subheader = st.subheader('Enter an API key in the sidebar to chat with your pdf.',divider=True)
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
    st.write(f'{store_name}')
    if os.path.exists(f"{store_name}.pkl"):
        with open(f"{store_name}.pkl", "rb") as f:
            VectorStore = pickle.load(f)
    else:
        
        embeddings=HuggingFaceEmbeddings()
        VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
        with open(f"{store_name}.pkl", "wb") as f:
            pickle.dump(VectorStore, f)
    return VectorStore

def main():
    """
    Main function to handle PDF text extraction, vector store processing, and user query handling.
    
    - Extracts text from a PDF file.
    - Processes the text and loads or creates a vector store.
    - Accepts user input for queries about the PDF.
    - Checks token count and performs a similarity search if within token limits.
    - Uses an LLM to generate a response based on the similarity search results.
    """

    text = extract_text_from_pdf(pdf)
    VectorStore = process_text_and_load_vector_store(text)
    query = st.text_input("Ask questions about your PDF file:")
    try:
        if query:
            docs = VectorStore.similarity_search(query=query, k=3)
            llm = OpenAI(openai_api_key=openai_api_key,temperature=0,)
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
            st.write(response)
    except AuthenticationError:
        st.warning(
            body='AuthenticationError : Please provide correct api key 🔑' ,icon='🤖')
    except Exception as e:
        st.warning(f"An error occure while processing the query: {e}")


if __name__ == '__main__':
    if openai_api_key:
        subheader.empty()
        pdf = st.file_uploader("Upload your PDF", type='pdf')
        if pdf:
            main()