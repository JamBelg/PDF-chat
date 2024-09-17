import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_pinecone import PineconeVectorStore
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain.chains.summarize import load_summarize_chain
import PyPDF2

def sideboard():
    st.sidebar.markdown(
    """
    <b>PDF file:</b>
    """, unsafe_allow_html=True)
    
    pdf_url = st.sidebar.text_input('URL to pdf file:', value="")
    pdf_file = st.sidebar.file_uploader('PDF file:', accept_multiple_files=False)
    
    st.sidebar.markdown(
    """
    ---
    """, unsafe_allow_html=True)
    
    OPENAI_API_KEY = st.sidebar.text_input('Add your OPENAI KEY:', value='', type='password')
    
    st.sidebar.markdown(
    """
    ---
    """, unsafe_allow_html=True)
    
    user_language = st.sidebar.selectbox("Translate to:", 
                                         ("English", "Spanish", "French", "German", "Italian", "Arabic"))
    
    st.sidebar.markdown(
    """
    ---
    
    **Developed by Jamel Belgacem**
    
    Connect with me:

    <a href="https://www.linkedin.com/in/jamel-belgacem-289606a7/" target="_blank">
        <img src="https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png" width="30" height="30" alt="LinkedIn"/>
    </a>
    
    <a href="https://github.com/JamBelg" target="_blank">
        <img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" width="30" height="30" alt="GitHub"/>
    </a>
    """,
    unsafe_allow_html=True
    )
    return OPENAI_API_KEY, pdf_url, pdf_file, user_language

def read_split_pdf(path):
    loader = PyPDFLoader(path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    pages = loader.load_and_split(text_splitter)
    
    return pages


@st.cache_resource
def initialize_model(pdf):
    # Check if the model is already loaded in session state
    if "model" not in st.session_state:
        # Initialize Model
        model_load_state = st.text('Loading model...')
        OPENAI_API_KEY = st.secrets['OPENAI_API_KEY']
        model = ChatOpenAI(openai_api_key=OPENAI_API_KEY,
                           model="gpt-3.5-turbo",
                           temperature=0.2)  # Use selected temperature
        # Embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        model_load_state.text("Model loaded!")

        # Documents
        data_load_state = st.text('Reading data...')
        documents = read_split_pdf(pdf)
        data_load_state.text("Data loaded!")

        # Prompt template (context + question)
        Rag_load_state = st.text('Preparing RAG ...')
        template = """
        Answer the question based on the content below. If you can't answer the question, reply 'I don't know'.
        Context: {context}
        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)
        
        # Parser
        parser = StrOutputParser()

        # VectorStore
        index_name = "pdf-files"
        PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
        os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
        pinecone = PineconeVectorStore.from_documents(
            documents,
            embedding=embeddings,
            index_name=index_name
        )
        Rag_load_state.text('RAG ready')

        # Save components to session state
        st.session_state.model = model
        st.session_state.embeddings = embeddings
        st.session_state.prompt = prompt
        st.session_state.parser = parser
        st.session_state.pinecone = pinecone
        st.session_state.documents = documents
    
    return bool(model and pinecone and documents and embeddings)

def model_retrieval(chat_history, retrieval, prompt, model, parser, language="English"):
    context = "\n".join([f"{message['role']}: {message['content']}" for message in chat_history])

    chain = (
        {"context": retrieval.as_retriever(), "question": RunnablePassthrough()}
        | prompt
        | model
        | parser
    )
    response = chain.invoke(context)

    if language != "English":
        translate_template = "Translate the following text to {language}: {{text}}".format(language=language)
        translate_prompt = ChatPromptTemplate.from_template(translate_template)
        translate_chain = translate_prompt | model
        translated_response = translate_chain.invoke(response)
        return translated_response.content

    return response

""" def summarize(model, documents):
    chain = load_summarize_chain(model, chain_type="stuff")
    result = chain.invoke(documents)
    return result """

def main():
    st.title("PDF Chatbot")

    st.markdown("<p style='font-size:16px;'>This Streamlit application uses LangChain 🦜 and OpenAI 🤖 to read and summarize PDFs</p>",
                unsafe_allow_html=True)
    st.markdown("<p style='font-size:16px;'>Users can interact with a chatbot and can translate the responses into multiple languages (🇬🇧 🇪🇸 🇫🇷 🇩🇪 🇸🇦)</p>",
                unsafe_allow_html=True)
    st.markdown("---", unsafe_allow_html=True)
    
    user_API_key, pdf_url, pdf_file, user_language = sideboard()

    if pdf_url:
        ind_init = initialize_model(pdf_url)
    elif pdf_file:
        ind_init = initialize_model(pdf_file)
    else:
        ind_init = False
    
    if ind_init:
        retrieval = st.session_state.pinecone
        prompt = st.session_state.prompt
        model = st.session_state.model
        parser = st.session_state.parser
        documents = st.session_state.documents
        
    st.markdown("<b>Chatbot</b> :robot_face:", unsafe_allow_html=True)
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_question := st.chat_input("How can I help you ?"):
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        if ind_init:
            response = model_retrieval(st.session_state.messages, retrieval, prompt, model, parser, user_language)
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)
        else:
            with st.chat_message("assistant"):
                st.markdown("Model not initialized or pdf missing!")

    


if __name__ == "__main__":
    main()
