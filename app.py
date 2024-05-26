import streamlit as st
import openai
import os
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables from .env file if available
load_dotenv()

# Set OpenAI API key
openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
else:
    openai.api_key = openai_api_key

# Load a pre-trained model for semantic similarity
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to get response from OpenAI using chat completions
def get_openai_response(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150
    )
    return response['choices'][0]['message']['content'].strip()

st.title("General Bot")

prompt = st.text_input("Enter your question here:")

if st.button("Get Response"):
    if prompt:
        if not openai_api_key:
            st.error("OpenAI API key is missing. Please set the OPENAI_API_KEY environment variable.")
        else:
            with st.spinner('Getting response...'):
                try:
                    openai_response = get_openai_response(prompt)
                    st.subheader("Response : ")
                    st.write(openai_response)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a question to get a response.")

st.sidebar.subheader("Disclaimer:")
st.sidebar.write("This chatbot is powered by OpenAI's GPT-4 model and Google's Gemini-Pro. The responses generated may not always be accurate or reliable. Use the information provided at your own discretion.")
