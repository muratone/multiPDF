# pip install streamlit pypdf2 langchain python-dotenv faiss-cpu openai huggingface_hub
# pip install InstructorEmbedding sentence_transformers
# pip install tiktoken
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
import time

# Free vs ChatOpenAI
from langchain.chat_models import ChatOpenAI
from langchain.llms import HuggingFaceHub
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings

# Does not work for Windows:
# from langchain.vectorstores import FAISS
# Does work for Windows:
# pip install chromadb==0.4.15
from langchain.vectorstores.chroma import Chroma

def get_pdf_text(pdf_docs):
    text = ""

    # from each pdf
    for pdf in pdf_docs: 
        pdf_reader = PdfReader(pdf)

        # loop all pages
        for page in pdf_reader.pages:
            text += page.extract_text()

    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorestore(text_chunks):
    st.session_state.limit_search += 1
    if st.session_state.limit_search == 3:
        with st.spinner("Cooldown..."):
            time.sleep(10 * 60)
        st.session_state.limit_search = 0
    # OpenAI
    embeddings = OpenAIEmbeddings(
        max_retries=2,
        request_timeout=120,
    )

    # Instructor:
    #embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    # this one changed for Windows:
    vectorstore = Chroma.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    # OpenAI vs Free
    llm = ChatOpenAI(
        max_retries=2,
        request_timeout=120,
        max_tokens=1600
    )
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    
    st.session_state.cooldown += 1
    if st.session_state.cooldown > 3:
        with st.spinner("Cooldown..."):
            time.sleep(10 * 60)
        st.session_state.cooldown = 0
    print(st.session_state.cooldown)
    st.session_state.cooldown += 1
    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response["chat_history"]

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
    st.session_state.wquestion = None

def clear_text():
    st.session_state.user_question = st.session_state.widget
    st.session_state.widget = ""

def button_pressed():
    st.session_state.wquestion = st.session_state.wbutton

def button_upload():
    st.session_state.pdf_docs = st.session_state.wfile
    st.session_state.user_question = None
    st.session_state.chat_history = None
    st.session_state.conversation = None


def main():
    load_dotenv()
    st.set_page_config(page_title="Multi-PDF-Chat", page_icon=":books:")

    footer = """
    <style>
    a:link , a:visited{
    color: white;
    background-color: transparent;
    text-decoration: underline;
    }

    a:hover,  a:active {
    color: red;
    background-color: transparent;
    text-decoration: underline;
    }

    .footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: black;
    color: white;
    text-align: center;
    }

    </style>
    <div class="footer">
    <p>Developed by <a href="mailto:kaidtheinternetuser@gmail.com">Kaid</a>
    </div>
    """

    st.markdown(footer, unsafe_allow_html=True) 
    # css
    st.write(css, unsafe_allow_html=True)

    # persistant variables
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "user_question" not in st.session_state:
        st.session_state.user_question = None
    if "wbutton" not in st.session_state:
        st.session_state.wquestion = None
        st.session_state.cooldown = 0
        st.session_state.limit_search = 0
    if "wfile" not in st.session_state:
        st.session_state.pdf_docs = None

    st.header("Multi PDF chat :books:")
    st.text_input("Ask a question about your documents:", on_change=clear_text, key="widget")
    if st.session_state.user_question:
        try:
            handle_userinput(st.session_state.user_question)
        except TypeError as e:
            pass

    with st.sidebar:
        st.subheader("Your documents")
        st.file_uploader("Upload your PDFs here and click on 'Process'", on_change=button_upload, accept_multiple_files=True, key="wfile")
        
        st.button("Process", on_click=button_pressed, key="wbutton")
        if st.session_state.wquestion:
            st.session_state.conversation = None
            with st.spinner("Processing..."):
                # get pdf
                if st.session_state.pdf_docs:
                    raw_text = get_pdf_text(st.session_state.pdf_docs)
            
                    # get text chunks
                    text_chunks = get_text_chunks(raw_text)

                    # create vector store
                    vectorstore = get_vectorestore(text_chunks)

                    # create converstation chain
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    st.session_state.wquestion = None
                
if __name__ == "__main__":
    main()
    