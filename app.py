from flask import Flask, render_template, request, redirect, url_for
from src.helper import download_huggingface_embeddings
from langchain.vectorstores import Pinecone
import pinecone
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains import RetrievalQA
from src.prompt import system_prompt
from dotenv import load_dotenv
from src.prompt import *
from langchain_openai import OpenAI
import os

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

embedding_model = download_huggingface_embeddings()

index_name = "medical-bot"

#existing index
docsearch = Pinecone.from_existing_index(
    index_name=index_name,
    embedding=embedding_model,
)

retriever = docsearch.as_retriever(search_type="similarity",search_kwargs={"k":3})
llm = OpenAI(temperature=0.4)

prompt = ChatPromptTemplate(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm,prompt)
rag_chain = create_retrieval_chain(retriever,question_answer_chain)


@app.route('/')
def index():
    return render_template('index.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"input": msg})
    # Extract just the answer content, removing any "System:" prefix
    clean_response = response["answer"].replace("System: ", "").strip()
    print("Response : ", clean_response)
    return str(clean_response)



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)