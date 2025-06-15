# Streamlit chat UI consuming the FastAPI /generate endpoint
# streamlit after uvicorn has been activated
# Run using streamlit run name.py

import requests
import streamlit as st

# Set the title of the Streamlit application
st.title("FastAPI ChatBot")

# Initialize chat messages in session_state if not already present
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing chat messages from session_state
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle new user input
if prompt := st.chat_input("Write your prompt in this input field"):
    # Append user message to session_state
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.text(prompt)

    # Fetch assistant response from FastAPI endpoint
    response = requests.get("http://localhost:8000/generate/text", params={"prompt": prompt})
    response.raise_for_status()  # Raise an error if the request was unsuccessful

    # Append assistant response to session_state
    st.session_state.messages.append({"role": "assistant", "content": response.text})

    # Display assistant message
    with st.chat_message("assistant"):
        st.markdown(response.text)