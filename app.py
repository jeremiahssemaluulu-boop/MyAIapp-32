import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()

st.title("Music Recommender")

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
    st.caption(f"{len(st.session_state.messages)} messages in history")
SYSTEM_PROMPT= """You are music recommender, you can only recommend music. 
The user will give you their preferences and you will recommend them a song based off those preferences.
If a user asks you for anything else you are not aloud to answer and you will redirect them to music."""

prompt = st.chat_input("Ask me something.. ")

for old in st.session_state.messages:
    with st.chat_message(old["role"]):
        st.markdown(old["content"])

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.getenv("GITHUB_TOKEN"),
    )
    with st.chat_message("user"):
        st.write(prompt)
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=model,
            temperature=creativity,
            messages=[{"role": "user", "content":SYSTEM_PROMPT}]+ st.session_state.messages[-recent_chat_memory:],
            stream=True
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