import os
from dotenv import load_dotenv
from backend.core import run_llm
import streamlit as st

load_dotenv()



def create_sources_string(sources: set) -> str:
    """Create a formatted string of source URLs."""
    if not sources:
        return ""
    sources_list = list(sources)
    sources_list.sort()
    sources_str = "sources:\n" 
    for i, src in enumerate(sources_list, start=0):
        sources_str = f"{i+1}. {src}\n"

    return sources_str

def main():
    print("Hello from ai-doc-assistant!")
    print(os.path.dirname(st.__file__))

    st.header("AI Langchain Document Assistant")
    prompt = st.text_input(label="Prompt", placeholder="Enter your question about LangChain:")

    ### streamlit has session_state to store variables across runs
    ### streamlit runs in an infinite loop and re-runs the script on every user interaction
    ### initialize these here
    if "user_prompt_history" not in st.session_state:
        st.session_state["user_prompt_history"] = []

    if "chat_answer_history" not in st.session_state:
        st.session_state["chat_answer_history"] = []

    if prompt:
        with st.spinner("Getting answer from AI..."):
            response = run_llm(prompt)
            # st.write("**Answer:**")
            # st.write(response["answer"])
            # st.write("**Source Documents:**")
            #convert to set to remove duplicates
            sources = set([doc.metadata["source"] for doc in response["source_documents"]])
            formatted_response = (
                f"{response['answer']}\n\n {create_sources_string(sources)}"
            )

            st.session_state["user_prompt_history"].append(prompt)
            st.session_state["chat_answer_history"].append(formatted_response)

    if st.session_state["chat_answer_history"]:
        for generated_response, user_query in zip(
            st.session_state["chat_answer_history"], st.session_state["user_prompt_history"]
        ):
            st.chat_message("User:").write(user_query)
            st.chat_message("AI:").write(generated_response)


if __name__ == "__main__":
    main()
