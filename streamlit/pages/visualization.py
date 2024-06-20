import streamlit as st

st.sidebar.page_link("app.py", label="Home", icon="🏠")
st.sidebar.page_link("pages/home.py", label="ChatBI", icon="💬")
st.sidebar.page_link("pages/pattern.py", label="Pattern Identifier", icon="📈")
st.sidebar.page_link("pages/visualization.py", label="Data Visualizer", icon="✨")
st.sidebar.page_link("pages/sqldata.py", label="Database Connector", icon="💽")



with st.container():
    st.title('Data Visualization')
    st.write('Here are some data visualization examples:')
    st.image('https://static.streamlit.io/examples/cat.jpg', caption='A cat', use_column_width=True)
    st.write('This is a cat.')
    st.write('This is just a test code We need to implement real code here.')