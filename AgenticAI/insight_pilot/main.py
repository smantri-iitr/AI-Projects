import streamlit as st
from claude import create_agent

if 'messages' not in st.session_state:
    st.session_state.messages = []

st.title('Insight Pilot')

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

if prompt := st.chat_input('Type your message here...'):
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    with st.chat_message('user'):
        st.markdown(prompt)

    response = create_agent(prompt)

    st.session_state.messages.append({'role': 'assistant', 'content': response})
    with st.chat_message('assistant'):
        st.markdown(response)
