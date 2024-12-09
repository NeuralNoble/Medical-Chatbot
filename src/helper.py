from langchain.document_loaders import PyPDFLoader , DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from sentence_transformers import SentenceTransformer


# Extract the Data from pdf file
def load_pdf_file(data):
    loader = DirectoryLoader(data,
                             glob="*.pdf",
                             loader_cls=PyPDFLoader)

    document = loader.load()
    return document

# Split the data into Text Chunks
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200,chunk_overlap=200)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks



# Download the Embeddings from the Hugging face
def download_huggingface_embeddings():
    embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MIniLM-L6-v2")
    return embeddings

