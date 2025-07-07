import streamlit as st

def initialize_session():
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "Hello! I am the SVL Chatbot. How can I assist you today?"}
        ]

def append_message(role: str, content: str):
    st.session_state["messages"].append({"role": role, "content": content}) 