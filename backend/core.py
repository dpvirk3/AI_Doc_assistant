from dotenv import load_dotenv
import os
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_pinecone import PineconeVectorStore
from typing import Any, Dict, List

# this takes two args: retriever object, 
#   and combine_docs_chain which has original imput, 
#           a context key with the retrieved documents
#                           and potentially a chat history
# this returns an LCEL runnable which returns a question and answer keys dictionary
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever

# hub is needed to dynamically download the augmentation prompt
from langchain import hub

# this chain combines documents to perform augmentation = takes in llm, 
# and a prompt template which has placeholder for 'context'
#    which is to be pass formatted documents to the llm
# OPTIONAL it can take an output_parser - defaults to strOutputParser
# OPTIONAL document_prompt - tells it to pick page_content key from the documents and metadate like url
# document_separator - the str separating two documents.
# returns a runnable (run by .invoke()) with a context key and any other inputs
#   invoke() on this runnable returns output depending on the output_parser
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

def run_llm (query: str, chat_history: List[Dict[str, Any]]=[]):
    embeddings = OllamaEmbeddings(model="jeffh/intfloat-e5-base-v2:f16", base_url="http://localhost:11434")
    # this is the retriever
    vectorstore = PineconeVectorStore(index_name=os.getenv("PINECONE_INDEX_NAME"),embedding=embeddings)

    chat = ChatOllama(model="gemma3:latest" ,verbose=True, temperature=0)

    #this is the dynamic prompt we use for RAG
    ## https://smith.langchain.com/hub/langchain-ai/retrieval-qa-chat
    ## it has placeholders {context} and {input}
    ## langchain plugs in the documents obtained via similarity search into {context}
    ##         and user query into {input}
    ## this prompt also has place for chat history as messages
    retriever_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    stuff_documents_chain = create_stuff_documents_chain(chat, retriever_qa_chat_prompt )

    # to use chat history we use a rephrase prompt
    ### link https://smith.langchain.com/hub/langchain-ai/chat-langchain-rephrase
    retreval_qa_rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
    history_aware_retriever = create_history_aware_retriever(
        llm=chat, retriever=vectorstore.as_retriever(),
        prompt=retreval_qa_rephrase_prompt
    )    
    
    # combine_docs_chain can be customized to simply plug all docs, or summarize them or whatever
    # qa = create_retrieval_chain (retriever = vectorstore.as_retriever(),
    #                              combine_docs_chain=stuff_documents_chain)
    ### to use chat history use history_aware_retriever
    qa = create_retrieval_chain (retriever = history_aware_retriever,
                                 combine_docs_chain=stuff_documents_chain)
    
    result = qa.invoke(input={"input": query, "chat_history": chat_history})

    new_results = {
        "query": result["input"],
        "answer": result["answer"],
        "source_documents": result["context"]
    }

    return new_results

def main():
    print("Hello from backend of the AIDocAssitant")
    res = run_llm("what is a langchain chain?")
    ### res is a dictionary with keys:
    ######   input  <-- user query
    ######   context  <-- augmentation docs
    ######   anwer <--- answer from the llm
    ###### we can see the urls as [doc.metadata["source"] for doc in res["context"]]
    print (res["answer"])


if __name__ == "__main__":
    main()