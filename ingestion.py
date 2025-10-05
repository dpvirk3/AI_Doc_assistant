import os
from dotenv import load_dotenv
import asyncio
import ssl
#to attach certificate to our https requests
import certifi
from  typing import Any, Dict, List
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
#langchain abstraction for text data with metadata for processing, embedding, indexing etc
from langchain_core.documents import Document
# to split documents into chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter
#TavilyCrawl is to get langchain documentation; the other two are for an optional video.
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap
# to use local index; we are using Pinecone cloud solution here
#from langchain_chroma import Chroma
#colorful logging - logger.py
from logger import Colors, log_info, log_success, log_error, log_warning, log_header


load_dotenv()

#configure ssl to use certifi certificates; needed for making Tavily API calls without certifcate errors
ssl_context = ssl._create_default_https_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

#chunk size tells how many documents/text objects langchain embeds in each call to ollamaEmbedding
#for very high number we may get rate limit error depending on token limit of the model
embeddings = OllamaEmbeddings(model="jeffh/intfloat-e5-base-v2:f16", base_url="http://localhost:11434")
vectorstore = PineconeVectorStore(index_name=os.getenv("PINECONE_INDEX_NAME"),embedding=embeddings)
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(map_depth=5, max_breadth=20, max_pages=1000)
tavily_crawl = TavilyCrawl()




async def main():
    """Async function to orchestrate document ingestion and indexing."""
    log_header("Starting Document Ingestion and Indexing Process")
    log_info("Loading documents from LangChain documentation website...",
             Colors.PURPLE)
    
    #tavily_crawl is a langchain tool so call invoke on it
    # recommended max_depth is 1 to 2, see results and change. 5 here is from Eden experience
    #extract_depth advanced ensures picking up tables and embedded text.
    # instructions is natural language filter for which pages to pick up
    res = tavily_crawl.invoke({"url":"https://python.langchain.com/docs/introduction/",
                             "max_depth" : "1",
                            "extract_depth" : "advanced",
                            "instructions": "content on ai agents"}) 
    

    # create langchain AI document with metadata from returned results
    all_docs = [Document (page_content=result["raw_content"],
                          metadata={"source": result["url"]})
                for result in res["results"]]
    log_success(f"Loaded {len(all_docs)} documents from LangChain documentation website")


if __name__ == "__main__":
    asyncio.run(main())