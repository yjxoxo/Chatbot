import streamlit as st
from langserve import RemoteRunnable

st.title('LangChain 기반 데이터 추천 애플리케이션')

user_input = st.text_input("어떤 데이터가 궁금하세요? 예)유동인구 데이터")

if st.button('전송'):
    remote = RemoteRunnable(url="http://localhost:8000/chat")
    result = remote.invoke({"input": user_input})
    st.write("응답:", result)