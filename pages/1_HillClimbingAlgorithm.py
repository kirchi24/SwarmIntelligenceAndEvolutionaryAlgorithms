import streamlit as st

st.title("Text Summarization Demo")

text = st.text_area("Enter text to summarize:")

if st.button("Summarize"):
    st.write("**Summary:**")
    st.write("→ This is where your summary would appear.")
