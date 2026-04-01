from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyPDFLoader , PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
import streamlit as st


load_dotenv()


if "document_load" not in st.session_state:
    st.session_state.document_load = False

if "agent" not in st.session_state:
    st.session_state.agent = None

if "vector_db" not in st.session_state:
    st.session_state.vector_db = False


if "messages" not in st.session_state:
    st.session_state.messages = []





# document loading 




def document_process(path):


    loader = PyPDFDirectoryLoader(path)
    loaded_doc = loader.load()



    # data splitting in chunks 

    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
    chunks = text_splitter.split_documents(loaded_doc)


    # chunks embedding
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


    # storing in Vector DataBase

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )




    @tool
    def RAG_one(query:str):
        """
        you are the pdf analyzer that return the answer from pdf ok in detail with detail explanation """

        similar_context = vector_store.similarity_search(query=query,k=3)

        context = ""
        for chunk in similar_context:
            context += chunk.page_content+'\n'
        

        return context






    system_promt = """
    so you are my assitance that give my all answer from global llm data and if not exist so you can search this thing in you data RAG or tools ok 
    and gives the answer in detail with the help of context  and with the help of question 
    """
    api_key = os.getenv("AI_KEY")
    model = ChatGroq(api_key=api_key,model="llama-3.3-70b-versatile")

    memory_saver = InMemorySaver()

    agent = create_agent(

                    model=model,
                    tools=[RAG_one],
                    system_prompt=system_promt,
                    checkpointer=memory_saver
                        )

    st.session_state.agent = agent
    st.session_state.document_load = True

        
    # 

    # 

    # res = model.invoke(f"your are my assistance that send me the answer by using context and and answer of the user ok so this is context:{context} and this is question of the user user:{query}")

    # print(res.content)  

if not st.session_state.document_load:

  uploaded = st.file_uploader(label="upload files", type=["pdf"],accept_multiple_files=True)
  if uploaded:
     with st.spinner(text="processing...."):
          path = "./data"
          for file in uploaded:
              with open(path + file.name , "wb" ) as f:
                   f.write(file.getvalue())

          document_process(path)   
          st.rerun()

    
   
if st.session_state.document_load and st.session_state.agent:
    
    if st.session_state.messages:
        for data in st.session_state.messages: 
           role = data.get("role")
           content = data.get("content")
           st.chat_message(role).markdown(content)

    query = st.chat_input("Ask about pdf")
    
    if query :
        st.chat_message("user").markdown(query)  

        st.session_state.messages.append({"role":"user","content":query})
        res = st.session_state.agent.invoke({"messages":[{"role":"user","content":query}]},{"configurable":{"thread_id":"1"}})
        result = res["messages"][-1].content
        st.session_state.messages.append({"role":"ai","content":result})
        st.chat_message("assistant").markdown(result)  # 👈 show instantly






