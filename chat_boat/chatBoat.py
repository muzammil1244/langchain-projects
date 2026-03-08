from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import streamlit as st
load_dotenv()

llm = os.getenv("GROQ_API")

model = ChatGroq(api_key=llm,model="llama-3.1-8b-instant")




st.title("Jarwis AI ChatBoat")
st.markdown("this an ai chatboat that help you ask anything whatever you want ")

if "messages" not in  st.session_state:
  st.session_state.messages = []


for data in st.session_state.messages:
    role = data["role"]
    content = data["content"]

    st.chat_message(role).markdown(content)

input_data = st.chat_input("ask any question")

if input_data:
  st.session_state.messages.append({"role":"user","content":input_data})
  st.chat_message("user").markdown(input_data)
  res = model.invoke(input_data)
  st.chat_message("ai").markdown(res.content)
  st.session_state.messages.append({"role":"ai","content":res.content})

