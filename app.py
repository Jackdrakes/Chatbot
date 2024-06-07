import streamlit as st
from bot import one_bit_retrieval_chain
st.write('# SCHEME BOT')


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask queries about Schemes and Initiatives?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream = "response"
        model_response = one_bit_retrieval_chain.invoke({"input": prompt})

        chat_response = f'''{model_response["answer"]}'''
        st.markdown(chat_response)
    st.session_state.messages.append({"role": "assistant", "content": chat_response})
