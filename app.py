import streamlit as st
import openai
import os
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from googlesearch import search
import requests
from bs4 import BeautifulSoup

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

# Function to perform a web search using googlesearch-python
def get_web_search_results(query):
    try:
        search_results = []
        for result in search(query, num_results=5):
            search_results.append(result)
        return search_results
    except Exception as e:
        return [f"Error during web search: {e}"]

# Function to scrape content from a URL
def scrape_content(url, max_paragraphs=5, max_chars=2000):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        content = ''
        char_count = 0
        for para in paragraphs[:max_paragraphs]:
            text = para.get_text()
            char_count += len(text)
            if char_count > max_chars:
                break
            content += text + ' '
        return content.strip()
    except Exception as e:
        return f"Error scraping {url}: {e}"

# Function to summarize content using GPT-4
def summarize_content(content):
    prompt = f"Summarize the following content:\n\n{content}"
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
                    # Get OpenAI response
                    openai_response = get_openai_response(prompt)
                    
                    # Get web search results
                    web_results = get_web_search_results(prompt)
                    
                    # Scrape and summarize content from top search results
                    summaries = []
                    for url in web_results:
                        content = scrape_content(url)
                        if content:
                            summary = summarize_content(content)
                            summaries.append((url, summary))
                    
                    st.subheader("Response from GPT-4")
                    st.write(openai_response)
                    
                    st.subheader("Web Search Results and Summaries")
                    for url, summary in summaries:
                        st.write(f"**Source:** {url}")
                        st.write(summary)
                        st.write("---")
                except openai.error.AuthenticationError as e:
                    st.error(f"OpenAI Authentication Error: {e}")
    else:
        st.warning("Please enter a question to get a response.")

st.sidebar.subheader("Disclaimer:")
st.sidebar.write("This chatbot is powered by OpenAI's GPT-4 model and real-time web searches. The responses generated may not always be accurate or reliable. Use the information provided at your own discretion.")
