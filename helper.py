from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List

def load_pdf_file(data: str) -> List[Document]:
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    return loader.load()

def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    return [Document(page_content=doc.page_content, metadata={"source": doc.metadata.get("source")}) for doc in docs]

def text_split(docs: List[Document], chunk_size: int = 500, chunk_overlap: int = 50) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)
