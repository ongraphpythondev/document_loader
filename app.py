import streamlit as st
from dotenv import load_dotenv
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

# Sidebar contents
    
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


load_dotenv()
try: 
    openai_api_key=st.secrets["OPENAI_API_KEY"]
    max_token_user=st.secrets["MAX_TOKEN_USER"]
    demo=st.secrets["DEMO"]#set demo 1
except:
    max_token_user=os.getenv("MAX_TOKEN_USER")
    openai_api_key=os.getenv('OPENAI_API_KEY')
    demo=os.getenv('DEMO')
def main():
    
    if demo:
        with st.sidebar:
            st.title('🤗💬 LLM Chat App')
            st.markdown('''
            ## About
            This app is an LLM-powered chatbot built using:
            - [Streamlit](https://streamlit.io/)
            - [LangChain](https://python.langchain.com/)
            - [OpenAI](https://platform.openai.com/docs/models) LLM model

            ''')
            add_vertical_space(5)
        st.header("Chat with PDF 💬")


        # upload a PDF file
        pdf = st.file_uploader("Upload your PDF", type='pdf')

        # st.write(pdf)
        if pdf is not None:
            pdf_reader = PdfReader(pdf)
            
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
                )
            chunks = text_splitter.split_text(text=text)

            # # embeddings
            store_name = pdf.name[:-4]
            st.write(f'{store_name}')
            # st.write(chunks)

            if os.path.exists(f"{store_name}.pkl"):
                with open(f"{store_name}.pkl", "rb") as f:
                    VectorStore = pickle.load(f)
                # st.write('Embeddings Loaded from the Disk')s
            else:
                # embeddings = OpenAIEmbeddings()
                
                embeddings=HuggingFaceEmbeddings()
                VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
                with open(f"{store_name}.pkl", "wb") as f:
                    pickle.dump(VectorStore, f)

            # embeddings = OpenAIEmbeddings()
            # VectorStore = FAISS.from_texts(chunks, embedding=embeddings)

            # Accept user questions/query
            query = st.text_input("Ask questions about your PDF file:")
            # st.write(query)
            user_token=num_tokens_from_string(query, "gpt-3.5-turbo")
            print("USER TOKEN COUNT: ", user_token)
            if query:
                if user_token < int(max_token_user):
                    docs = VectorStore.similarity_search(query=query, k=3)

                    llm = OpenAI(openai_api_key=openai_api_key,temperature=0,)
                    chain = load_qa_chain(llm=llm, chain_type="stuff")
                    

                    with get_openai_callback() as cb:
                        response = chain.run(input_documents=docs, question=query)
                        print(cb)
                    st.write(response)
                else:
                    st.write(f"EXCEED ALLOCATED PROMPT,\n MAX TOKEN: {max_token_user} \n YOUR TOKEN: {user_token}")

    else:
        st.header("This App is Private!!!")

if __name__ == '__main__':
    main()