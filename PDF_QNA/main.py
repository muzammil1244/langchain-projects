from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
load_dotenv()

# document loading 

document = "./data/resume.pdf"
loader = PyPDFLoader(document)
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

while True:
    question = input("ask question about you pdf:")
    searched_data = vector_store.similarity_search(query=question,k=2)

    context = ""

    for data in searched_data:
     context+= data.page_content+'\n'

# llm implementation

    api_key = os.getenv("AI_KEY")

    model = ChatGroq(api_key=api_key,model="llama-3.3-70b-versatile")

    res = model.invoke(f"your are my assistance that send me the answer by using context and and answer of the user ok so this is context:{context} and this is question of the user user:{question}")

    print(res.content)  
