from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent
import streamlit  as st
load_dotenv()


groq_api = os.getenv("GROQ_API")
model = ChatGroq(api_key=groq_api , model="llama-3.3-70b-versatile")



db  = SQLDatabase.from_uri("sqlite:///SQL_DB.db")


db.run("""
    CREATE TABLE IF NOT EXISTS todo(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    description TEXT,
    status TEXT CHECK( status IN ('pending' , 'work_in_progress','completed')) DEFAULT 'pending',
    create_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
)


toolkit = SQLDatabaseToolkit(db=db , llm=model)
tools = toolkit.get_tools()
memory = InMemorySaver()
system_prompt = """

You are a task management assistant that interacts with a SQL database containing a 'todo' table. 

TASK RULES:
1. Limit SELECT queries to 10 results max with ORDER BY create_at DESC
2. After CREATE/UPDATE/DELETE, confirm with SELECT query
3. If the user requests a list of todo, present the output in a structured table format to ensure a clean and organized display in the browser."

CRUD OPERATIONS:
    CREATE: INSERT INTO todo(title, description, status)
    READ: SELECT * FROM todo WHERE ... LIMIT 10
    UPDATE: UPDATE todo SET status=? WHERE id=? OR title=?
    DELETE: DELETE FROM todo WHERE id=? OR title=?

Table schema: id, title, description, status(pending/work_in_progress/completed), create_at.
"""


# agent code
st.cache_resource
def call_agent():
  agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=system_prompt,
    checkpointer=memory

   )
  return agent

if "history" not in st.session_state:
  st.session_state.history = []




for data in st.session_state.history:
  role = data["role"]
  content = data["content"]
  st.chat_message(role).markdown(content)



agent =call_agent()
st.subheader(" TODO TASK MANAGER ")
user_input = st.chat_input("perform operation on todo db : ")

if user_input:
   st.session_state.history.append({"role":"user","content":user_input})
   st.chat_message("user").markdown(user_input) 


   with st.chat_message("ai"):
      with st.spinner("progress...."):
       res = agent.invoke({
      "messages":{"role":"user","content":user_input}
   }, {"configurable":{"thread_id":"1"}})
       
       result = res["messages"][-1].content
       st.markdown(result)
       st.session_state.history.append({"role":"ai","content":result})











# while True :
#     user_input = input("User :")
#     res = agent.invoke(
#         {"messages":[{"role":"user","content":user_input}]},
#         {"configurable":{"thread_id":"1"}}
#     )


#     print("AI:"+ res["messages"][-1].content)