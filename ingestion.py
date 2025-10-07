import os
from dotenv import load_dotenv
import asyncio
import ssl
#to attach certificate to our https requests
import certifi
from  typing import Any, Dict, List
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
#from pinecode import Pinecone
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
from tenacity import retry, wait_random_exponential, stop_after_attempt


load_dotenv()

BASE_URL_TO_CRAWL = os.getenv("BASE_URL_TO_CRAWL")

#configure ssl to use certifi certificates; needed for making Tavily API calls without certifcate errors
ssl_context = ssl._create_default_https_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

#chunk size tells how many documents/text objects langchain embeds in each call to ollamaEmbedding
#for very high number we may get rate limit error depending on token limit of the model
embeddings = OllamaEmbeddings(model="jeffh/intfloat-e5-base-v2:f16", base_url="http://localhost:11434")
vectorstore = PineconeVectorStore(index_name=os.getenv("PINECONE_INDEX_NAME"),embedding=embeddings)
#pinecone_env = Pinecone(api_key=os.getenv("PINECONE_API_KEY"), environment="us-east1-aws")
#pinecone_index = pinecone_env.index(os.getenv("PINECONE_INDEX_NAME"))

tavily_extract = TavilyExtract()
tavily_map = TavilyMap(map_depth=5, max_breadth=20, max_pages=1000)
tavily_crawl = TavilyCrawl()

def chunk_urls (urls: List[str], chunk_size: int = 5) -> List[List[str]]:
    """Utility function to chunk a list of URLs into smaller lists of a specified size."""
    chunks=[]
    for i in range(0, len(urls), chunk_size):
        chunk = urls[i:i + chunk_size]
        chunks.append(chunk)
    
    return chunks

async def extract_text_from_urls(urls: List[str], batch_num: int) -> List[Dict[str, Any]]:
    """Extract text content from a list of URLs using TavilyExtract tool."""

    extracted_results = []
    try:
        log_info(f"Processing batch num: {batch_num}", Colors.DARKCYAN)
        docs = await tavily_extract.ainvoke({"urls": urls})
        extracted_results = docs.get("results", [])
        log_success(f"Extracted text from {len(extracted_results)} URLs in batch {batch_num}")
        
    except Exception as ex:
        log_error(f"Error extracting text from URLs in batch {batch_num}: {ex}")
        
    return extracted_results



async def async_extract(url_batch: List[List[str]]):
    """Asynchronously extract text from batches of URLs."""
    log_header("Starting asynchronous extraction of text from URL batches")    

    tasks = [extract_text_from_urls(batch, i+1) for i, batch in enumerate(url_batch, start=0)]
    all_results = await asyncio.gather(*tasks, return_exceptions=True)
    
    all_raw_content = []
    failed_batches = 0
    for result in all_results:
        if isinstance(result, Exception):
            failed_batches += 1
            log_error(f"Batch failed with exception: {result}")
        else:
            for extracted_page in result:
                document = Document(page_content=extracted_page["raw_content"],
                                    metadata={"source": extracted_page["url"]})
                all_raw_content.append(document)
    
    log_success(f"Successfully extracted text from {len(all_raw_content)} URLs with {failed_batches} failed batches")
    return all_raw_content

# Create a retryable upsert function
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def retry_upsert(index, vectors):
    index.upsert(vectors=vectors)

#coroutine to converted chunks into index
async def index_documents_async(docs: List[Document], batch_size: int = 50):
    """Asynchronously index documents in batches to the vector store."""
    log_header("Starting asynchronous indexing of documents to vector store")
    total_num_docs = len(docs)
    log_info(f"Total documents to index: {total_num_docs}", Colors.DARKCYAN)

    #create a list of batches of documents
    batches = [docs[i:i + batch_size] for i in range(0, total_num_docs, batch_size)]
    log_info(f"Created {len(batches)} batches for indexing", Colors.DARKCYAN)

    #coroutine inside this coroutine to submit batches
    async def index_batch(batch: List[Document], batch_num: int):
        try:
            log_info(f"Indexing batch {batch_num} with {len(batch)} documents", Colors.DARKCYAN)
            #### aadd - asynchronous adding to pinecone
            await vectorstore.aadd_documents(batch)
            #await pinecone_index.aupsert_documents(batch)
            log_success(f"Successfully indexed batch {batch_num}")
        except Exception as ex:
            log_error(f"Error indexing batch {batch_num}: {ex}")
            return False
        
        return True
    
    tasks = [index_batch(batch, i+1) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    #count successes and failures
    success_count = sum(1 for r in results if r is True)
    if success_count == len(batches):
        log_success("All batches indexed successfully.")
    else:
        log_warning(f"{len(batches) - success_count} out of {len(batches)} batches failed to index.")   


    log_success("Completed indexing all documents.")

async def main():
    """Async function to orchestrate document ingestion and indexing."""
    log_header("Starting Document Ingestion and Indexing Process")
    log_info("Loading documents from LangChain documentation website...",
             Colors.PURPLE)
    
    if  os.getenv("TAVILY_USE_BATCH_PROCESSING").lower() == "true":
        log_info("Using Tavily batch processing for crawling and extracting web pages",
                 Colors.PURPLE)
        site_map = tavily_map.invoke({"url":BASE_URL_TO_CRAWL})

        log_success(f"Site map created with {len(site_map['results'])} pages to extract")
        url_chunks = chunk_urls(site_map["results"], chunk_size=5)

        log_info(f"Processing {len(url_chunks)} chunks of URLs for extraction", Colors.PURPLE)

        all_docs = await async_extract(url_chunks)
        
        log_success(f"Extracted text from {len(all_docs)} URLs from LangChain documentation website")       

    else:
        log_info("Using Tavily synchronous processing for crawling and extracting web pages",
                 Colors.PURPLE)
        #tavily_crawl is a langchain tool so call invoke on it
        # recommended max_depth is 1 to 2, see results and change. 5 here is from Eden experience
        #extract_depth advanced ensures picking up tables and embedded text.
        # instructions is natural language filter for which pages to pick up
        res = tavily_crawl.invoke({"url":BASE_URL_TO_CRAWL,
                                "max_depth" : "1",
                                "extract_depth" : "advanced",
                                "instructions": "content on ai agents"}) 
    

        # create langchain AI document with metadata from returned results
        all_docs = [Document (page_content=result["raw_content"],
                            metadata={"source": result["url"]})
                    for result in res["results"]]
        log_success(f"Loaded {len(all_docs)} documents from LangChain documentation website")

    #chunk documents using text splitter
    log_header("Splitting documents into smaller chunks for embedding")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    splitted_docs = text_splitter.split_documents(all_docs)
    log_success(f"Split into {len(splitted_docs)} chunks of text from {len(all_docs)} documents")

    #index documents in batches asynchronously
    await index_documents_async(splitted_docs, batch_size=100)

    log_header("INDEXING PIPELINE COMPLETED SUCCESSFULLY")
    log_info("SUMMARY:", Colors.BOLD)
    log_info(f" URLs mapped: {len(site_map['results'])}")
    log_info(f" Documents extracted: {len(all_docs)}")
    log_info(f" Text chunks created: {len(splitted_docs)}")
    log_info(f" Vector store: Pinecone - {os.getenv('PINECONE_INDEX_NAME')}")

if __name__ == "__main__":
    asyncio.run(main())