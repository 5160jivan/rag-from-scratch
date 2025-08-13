from dotenv import load_dotenv
from langchain.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import bs4
from langchain.load import dumps, loads
import numpy as np

import os

# Load environment variables from .env file
load_dotenv()


def get_post_retriever():
    loader: WebBaseLoader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    docs: List[Document] = loader.load()

    # Split
    text_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits: List[Document] = text_splitter.split_documents(docs)

    # Embed
    vectorstore: Chroma = Chroma.from_documents(documents=splits, 
                                            embedding=OpenAIEmbeddings())

    retriever = vectorstore.as_retriever()
    return retriever

def get_unique_union(documents: list[list]):
    """ Unique union of retrieved docs """
    # Flatten list of lists, and convert each Document to string
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    # Get unique documents
    unique_docs = list(set(flattened_docs))
    # Return
    return [loads(doc) for doc in unique_docs]

def cosine_similarity(vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        return dot_product / (norm_vec1 * norm_vec2)


