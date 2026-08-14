import os
import dotenv
import openai
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()
import chromadb
import chromadb.utils.embedding_functions as ef
db = chromadb.PersistentClient(path="./chroma_db")
memories = db.get_or_create_collection("my_facts")
memories.upsert(
documents=[
"My name is Jeremiah",
"I play video games",
"My favorite food is hamburgers"
],
ids=["fact4", "fact5", "fact6"],
)
print("\nstored:", memories.count(), "facts")

question = "What is the best food?"

results = memories.query(query_texts=[question], n_results=4)
notes= "\n".join(results["documents"][0])


prompt = f"Using these notes: {notes} \n\n  {question}"
print(prompt)
client = OpenAI(
base_url="https://api.groq.com/openai/v1",
api_key=os.getenv("GITHUB_TOKEN"),
)
r = client.chat.completions.create(
model="llama-3.3-70b-versatile",
messages=[{"role": "user", "content": prompt}],
)
# print(r) # uncomment to see the whole messy response
print(r.choices[0].message.content)
