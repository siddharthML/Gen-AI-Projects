# Create streamlit app for audio UI
# Run using streamlit run name.py

import requests
import streamlit as st

# Set the title of the Streamlit app
st.title("FastAPI ChatBot")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []  # Create an empty list to store messages

# Display existing messages
for message in st.session_state.messages:
    # Create a chat message container for each message
    with st.chat_message(message["role"]):
        content = message["content"]
        # Check if the content is audio (byte data)
        if isinstance(content, bytes):
            st.audio(content)  # Display audio content
        else:
            st.markdown(content)  # Display text content

# Input for new prompt
if prompt := st.chat_input("Write your prompt in this input field"):
    # Send a request to the FastAPI server to generate audio
    response = requests.get(
        f"http://localhost:8000/generate/audio", params={"prompt": prompt}
    )
    response.raise_for_status()  # Raise an exception for HTTP errors
    # Display the generated audio
    with st.chat_message("assistant"):
        st.text("here is your generated audio")
        st.audio(response.content)  # Display audio content
