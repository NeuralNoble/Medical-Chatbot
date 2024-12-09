from src.helper import load_pdf_file,text_split,download_huggingface_embeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_community.vectorstores import Pinecone
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

extracted_data = load_pdf_file("Data/")
text_chunks = text_split(extracted_data)
embedding_model = download_huggingface_embeddings()


pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medical-bot"

pc.create_index(
    name=index_name,
    dimension=384,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1",
    )
)

docsearch = Pinecone.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embedding_model,
)

