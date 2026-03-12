from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import streamlit as st
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_community.utilities import GoogleSerperAPIWrapper
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

llm = os.getenv("GROQ_API")
serper = os.getenv("SERPER_API")
model = ChatGroq(model="llama-3.1-8b-instant",api_key=llm , streaming=True)
search = GoogleSerperAPIWrapper( serper_api_key=serper)


tools = [search.run]



memory = MemorySaver()

if "memory" is not st.session_state:
    st.session_state.memory = MemorySaver()




agetn1 = create_agent(
    model=model ,
    tools=tools,
    system_prompt="you are smart ai chat boat that first answering from the llm data and if information is not in lmm then call tool ok ",
    checkpointer=st.session_state.memory


                    )




st.title(" AI Cha boat ")


if "messages"  not in st.session_state:
    st.session_state.messages = []



input_chat = st.chat_input("how can i help you  ? : ")


for message in st.session_state.messages:
    role = message["role"]
    mess = message["content"]
    st.chat_message(role).markdown(mess)


if input_chat:
    st.chat_message("user").markdown(input_chat)
    st.session_state.messages.append(
    {"role":"user","content":input_chat}
)
    
    res = agetn1.stream(
    {"messages": st.session_state.messages},
    {"configurable":{"thread_id":"1"}},
    stream_mode="messages"
)
    
    ai_container = st.chat_message("ai")
    with ai_container:
         space = st.empty()
         message = ""

         for chunck in res:
             message = message + chunck[0].content
             space.write(message)


         st.session_state.messages.append({"role":"ai" , "content":message})
