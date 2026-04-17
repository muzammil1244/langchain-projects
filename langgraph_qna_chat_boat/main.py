from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langgraph.graph import StateGraph , START , END
from pydantic import BaseModel
import os
from typing import Annotated
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
# ///

load_dotenv()

llm = os.getenv("GROQ_API")

# model

model = ChatGroq(api_key=llm , model="llama-3.3-70b-versatile")

# class

class ChatState(BaseModel):
    messages:Annotated[list ,add_messages]

# graph work 


graph = StateGraph(ChatState)


def Chat_function(state:ChatState) -> ChatState:

    query = state.messages
    res = model.invoke(query)
    state.messages = [res]
    return state

# nodes

graph.add_node("chatfun",Chat_function)

graph.add_edge(START,"chatfun")
graph.add_edge("chatfun",END)

memory = InMemorySaver()

final_graph = graph.compile(checkpointer=memory)


while True:

    query = input("Ask something ! :")
    
    if query.lower() in ["exit","done"]:
         print("thanks for connecting")
         break

    res = final_graph.invoke(
        {
            "messages":[
            {
                "role":"user","content":query
            }
            ]
            }
            ,{"configurable":{"thread_id":"a1"}}
            )
    print(res["messages"][-1].content)