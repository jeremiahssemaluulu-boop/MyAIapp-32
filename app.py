import streamlit as st
st.title("My First AI")

st.write("anything you want")

st.header("Header")
st.subheader("Subheader")
count=0

name=st.text_input("Enter your name")

if st.button("Submit"):
    st.write(f"Hello {name}!")

st.selectbox("Select a number", [1,2,3,4,5])
st.slider("Select a slider", min_value=0, max_value=10)

rm - rf.git
git
init
git
add.
git
commit - m
"first commit"
git
branch - M
main
git
remote
add
origin[https://github.com/jeremiahssemaluulu-boop/MyAIapp-32.git]
git
push - u
origin
main - -force