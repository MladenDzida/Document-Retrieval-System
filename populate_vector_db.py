import gdown
from langchain_core.documents import Document
from langchain_community.document_loaders import DataFrameLoader
from langchain_text_splitters import CharacterTextSplitter
import uuid
import chromadb
from chromadb.config import Settings
import zipfile
import pandas as pd
from typing import List

def download_and_extract_data() -> None:
    """
    Download and extract data from a Google Drive link.

    This function downloads a zip file from a given Google Drive link and extracts its contents
    into the current directory.
    """
    gdown.download('https://drive.google.com/uc?id=1KB6gCv2aTc1DOBF1RoEVhqFHfBL4_ZMn', './data.zip', quiet=False)
    with zipfile.ZipFile('./data.zip', 'r') as zip_ref:
        zip_ref.extractall('.')

def load_and_split_documents(df: pd.DataFrame, chunk_size: int = 128, chunk_overlap: int = 32) -> List[Document]:
    """
    Load documents from a DataFrame and split them into smaller chunks.

    This function takes a DataFrame containing documents, loads the documents, and splits them
    into smaller chunks based on the specified chunk size and overlap. This is useful for processing
    large documents in smaller, more manageable pieces.

    Args:
        df (pd.DataFrame): The DataFrame containing the documents to be loaded and split.
        chunk_size (int, optional): The size of each chunk. Default is 128 characters.
        chunk_overlap (int, optional): The number of characters to overlap between chunks. Default is 32 characters.

    Returns:
        List[Document]: A list of Document objects representing the split chunks.
    """
    loader = DataFrameLoader(df)
    data = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(data)
    return docs

def populate_vector_db() -> None:
    """
    Populate a ChromaDB vector store with documents from a CSV file.

    This function downloads and extracts data from a predefined source, loads and splits the documents
    from the extracted CSV file, and populates a ChromaDB vector store with these documents.
    """
    CHROMADB_HOST = 'host.docker.internal'
    CHROMADB_PORT = 8000
    CHROMADB_SETTINGS = Settings(allow_reset=True)

    download_and_extract_data()

    # Load the data
    df = pd.read_csv('blogtext_small.csv', delimiter=',')

    docs = load_and_split_documents(df=df)

    # Init chromadb client and collection
    client = chromadb.HttpClient(host=CHROMADB_HOST, port=CHROMADB_PORT, settings=CHROMADB_SETTINGS)
    client.reset()  # resets the database
    collection = client.get_or_create_collection("blog_collection")

    # add the data to the collection
    for doc in docs:
        collection.add(
            ids=[str(uuid.uuid1())], metadatas=doc.metadata, documents=doc.page_content
        )


if __name__ == '__main__':
    populate_vector_db()