import streamlit as st 
from app import create_vector_db, get_qa_chain


st.title("Guide du Code du Travail Marocain")
btn=st.button("Create knowledgebase")


if btn :
    pass

question = st.text_input("question : ")

if question:
    chain = get_qa_chain()
    response = chain(question)

    st.header("Answer")
    st.write(response["result"])