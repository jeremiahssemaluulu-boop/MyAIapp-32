import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()

import chromadb
from doc_helper import read_file

st.title("Music Recommender")

db= chromadb.PersistentClient(path="./chroma_db")
brain = db.get_or_create_collection("docs")

def chunk_it(text, size=400):
    bits = text.split(". ")
    chunks, current = [], ""
    for bit in bits:
        if len(current) + len(bit) < size:
            current += bit + ", "
        else:
            if current.strip():
                chunks.append(current.strip())
            current = bit + ", "
    if current.strip():
        chunks.append(current.strip())
    return chunks

def store_document(file):
    chunks = chunk_it(read_file(file))
    prefix = file.name.replace(" ", "_")
    brain.upsert(
        documents = chunks,
        ids = [f"{prefix}_{i}" for i in range(len(chunks))],
    )
    return len(chunks)

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Settings")
    name = st.text_input("Enter your name")
    recent_chat_memory = st.slider("Recent chat memory", 2,15,5)
    mood = st.selectbox("What will your AI's mood be?", ["Happy", "Sad", "Angry"])
    model = st.selectbox("model",["openai/gpt-oss-120b", "openai/gpt-oss-120b"])
    creativity = st.slider("Creativity", 0.0, 1.0, 0.3)
    if st.button("Clear chat history"):
        st.session_state.messages = []
        st.rerun()

    if st.button("Forget Documents"):
        db.delete_collection("docs")
        st.rerun()
    st.caption(f"{len(st.session_state.messages)} messages have been sent in this chat")
    st.caption(f"{brain.count()} chunks stored from your documents")
SYSTEM_PROMPT= """You are music recommender, you can only recommend music. 
The user will give you their preferences and you will recommend them a song based off those preferences.
If a user asks you for anything else you are not aloud to answer and you will redirect them to music."""


for old in st.session_state.messages:
    with st.chat_message(old["role"]):
        st.markdown(old["content"])
user_input = st.chat_input("Ask me something in here.. ", accept_file=True, file_type=["pdf", "text"])
if user_input:
    prompt = user_input.text
    if user_input.files:
        n = store_document(user_input.files[0])
        st.success(f"Stored {user_input.files[0].name} as {n} chunks")

if user_input and prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1/",
        api_key=os.getenv("GITHUB_TOKEN")
    )
    with st.chat_message("user"):
        st.write(prompt)
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=model,
            temperature=creativity,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + st.session_state.messages[-recent_chat_memory:],
            stream=True,
        )

    thinking = st.expander("Thinking...")
    answer = st.empty()
    t = a = ""
    for chunk in stream:
        d = chunk.choices[0].delta
        if getattr(d, "reasoning", None):
            t += d.reasoning
            thinking.markdown(f"*{t}*")
        if d.content:
            a += d.content
            answer.markdown(a)
    st.session_state.messages.append({"role": "assistant", "content": a})
    st.rerun()