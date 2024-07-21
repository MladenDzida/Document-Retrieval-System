from langchain_core.documents import Document
from langchain_community.document_loaders import DataFrameLoader
from langchain_text_splitters import CharacterTextSplitter
import pandas as pd
from typing import List
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

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

    This function loads and splits the documents from a predefined CSV file, and populates a ChromaDB
    vector store with these documents. The documents are split into chunks to manage memory usage
    and are processed in batches.
    """
    # Load the data
    df = pd.read_csv('blogtext_small.csv', delimiter=',')

    docs = load_and_split_documents(df=df)
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    def split_list(input_list, chunk_size):
        for i in range(0, len(input_list), chunk_size):
            yield input_list[i:i + chunk_size]

    split_docs_chunked = split_list(docs, 5000)

    for split_docs_chunk in split_docs_chunked:
        vectordb = Chroma.from_documents(
            documents=split_docs_chunk,
            embedding=embedding_function,
            persist_directory="./chroma_db",
        )
        vectordb.persist()


if __name__ == '__main__':
    populate_vector_db()