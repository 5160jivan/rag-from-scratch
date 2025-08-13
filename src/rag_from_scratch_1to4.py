from typing import List
from unittest import loader
import tiktoken
import numpy as np
import os
import bs4
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from langsmith.wrappers import wrap_openai
from langsmith import traceable


from dotenv import load_dotenv
import os

from utils import cosine_similarity

# Load environment variables from .env file
load_dotenv()

#### INDEXING ####


def rag_overview():
    # Load Documents
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

    #### RETRIEVAL and GENERATION ####

    # Prompt
    prompt = hub.pull("rlm/rag-prompt")

    # LLM
    llm: ChatOpenAI = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # Post-processing
    def format_docs(docs: List[Document]) -> str:
        return "\n\n".join(doc.page_content for doc in docs)

    # Chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Question
    resp: str = rag_chain.invoke("What is Task Decomposition?")
    print(resp)

def rag_documents():
    question = "What kinds of pets do I like?"
    document = "My favorite pet is a cat."

    def num_tokens_from_string(string: str, encoding_name: str) -> int:
        """Returns the number of tokens in a text string."""
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens

    
    embd = OpenAIEmbeddings()
    query_result = embd.embed_query(question)
    
    document_result = embd.embed_query(document)
    similarity = cosine_similarity(query_result, document_result)
    print("Cosine Similarity:", similarity)

def rag_indexing():
    loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
    )
    blog_docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=300, 
    chunk_overlap=50)

    # Make splits
    splits = text_splitter.split_documents(blog_docs)

   # Create vectorstore
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})


    docs = retriever.get_relevant_documents("What is Task Decomposition?")

    print("Relevant Document:", docs)
    return retriever, docs


# print hello world
def rag_generation():
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)
    llm = (ChatOpenAI(model="gpt-3.5-turbo", temperature=0))
    # chain = prompt | llm
    retriever, docs = rag_indexing()
    # chain.invoke({"context":docs,"question":"What is Task Decomposition?"})
    prompt_hub_rag = hub.pull("rlm/rag-prompt")
    rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

    resp = rag_chain.invoke("What is Task Decomposition?")
    print(resp)


rag_generation()
