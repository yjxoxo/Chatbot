import streamlit as st
from langserve import RemoteRunnable

st.title('LangChain ��� ������ ��õ ���ø����̼�')

user_input = st.text_input("� �����Ͱ� �ñ��ϼ���? ��)�����α� ������")

if st.button('����'):
    remote = RemoteRunnable(url="http://localhost:8000/chat")
    result = remote.invoke({"input": user_input})
    st.write("����:", result)