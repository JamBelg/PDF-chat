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

def sideboard():
    st.sidebar.markdown(
    """
    ---
    
    <u>About:</u>\
    
    This chatbot uses RAG to respond from specified PDF file.\
    
    ---
    
    <h3>Model settings:</h3>
    """, unsafe_allow_html=True)
    
    # Add a slider input for temperature settings
    user_temperature = st.sidebar.number_input('Select model temperature',
                                       min_value = 0.0,
                                       max_value = 1.0,
                                       value = 0.2,
                                       step = 0.1)
    

    OPENAI_API_KEY = st.sidebar.text_input('Add your OPEN AI KEY:',
                                           value='None',
                                           type='password')
    
    st.sidebar.markdown(
    """
    ---
    
    <b>PDF file path:</b>
    """, unsafe_allow_html=True)
    
    pdf_path = st.sidebar.text_input('Add url to pdf file:',
                                     value="https://arxiv.org/pdf/2402.16893")
    
    user_language = st.sidebar.selectbox("Translate to:",
                                 ("English","Spanich", "French", "German"))
    
    st.sidebar.markdown(
    """
    ---
    
    **Developped by Jamel Belgacem**
    
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
    return user_temperature, OPENAI_API_KEY, pdf_path, user_language

def read_split_pdf(path):
    loader = PyPDFLoader(file_path=path, extract_images=False)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=50)
    pages = loader.load_and_split(text_splitter)
    
    return pages

def initialize_model(temperature, API_key, pdf_path):
    # Check if the model is already loaded in session state
    if "model" not in st.session_state:
        # Initialize Model
        if API_key=='None':
            OPENAI_API_KEY = st.secrets['OPENAI_API_KEY']
        else:
            OPENAI_API_KEY=API_key
            
        model = ChatOpenAI(openai_api_key=OPENAI_API_KEY,
                           model="gpt-3.5-turbo",
                           temperature=temperature)
        # Embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

        # Documents
        documents = read_split_pdf(path = pdf_path)

        # Prompt template (context + question)
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

        # Save components to session state
        st.session_state.model = model
        st.session_state.embeddings = embeddings
        st.session_state.prompt = prompt
        st.session_state.parser = parser
        st.session_state.pinecone = pinecone
        st.session_state.documents = documents

def model_retrieval(chat_history, retrieval, prompt, model, parser, language="English"):
    # Combine chat history to provide context to the model
    context = "\n".join([f"{message['role']}: {message['content']}" for message in chat_history])
    
    # Response
    chain = (
        {"context": retrieval.as_retriever(), "question": RunnablePassthrough()}
        | prompt
        | model
        | parser
    )
    response = chain.invoke(context)
    
    # If the language parameter is not English, translate the response
    if language != "English":
        
        # Define the translation prompt template
        translate_template = "Translate the following text to {language}: {{text}}".format(language=language)
        translate_prompt = ChatPromptTemplate.from_template(translate_template)
        
        # Create a translation chain
        translate_chain = translate_prompt | model
        
        # Invoke translation chain
        translated_response = translate_chain.invoke(response)
        return translated_response.content
    
    return response

def summarize(model, documents):
    chain = load_summarize_chain(model, chain_type="stuff")
    result = chain.invoke(documents)
    return(result)

def main():
    st.title("PDF Chatbot")
    st.markdown("<h2>Ask questions about your PDF and receive your answer :) </h2>",
                unsafe_allow_html=True)
    
    user_temperature, user_API_key, pdf_path, user_language = sideboard()
    
    # Initialize the model and components
    state_mode = st.text("Model initialization...")
    initialize_model(user_temperature,
                     user_API_key,
                     pdf_path)
    state_mode.text("Model ready")

    # Retrieve components from session state
    retrieval = st.session_state.pinecone
    prompt = st.session_state.prompt
    model = st.session_state.model
    parser = st.session_state.parser
    documents = st.session_state.documents

    # Summarize
    summary = summarize(model=model, documents=documents)
    summary=summary["output_text"].replace(". ", ".\n")
    st.markdown("""
                ---
                <h3>Summary:</h3>""",
                unsafe_allow_html=True)
    st.markdown(summary, unsafe_allow_html=True)
    st.markdown("---", unsafe_allow_html=True)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if user_question := st.chat_input("What is up?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_question})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(user_question)
        

        # Generate response with chat history context
        response = model_retrieval(st.session_state.messages,
                                   retrieval,
                                   prompt,
                                   model,
                                   parser,
                                   user_language)
    
        # Add response message to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)

if __name__ == "__main__":
    main()
