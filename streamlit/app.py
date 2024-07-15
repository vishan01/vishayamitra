
import os
import pandas as pd
from pandasai import Agent
from pandasai.responses.response_parser import ResponseParser
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import streamlit.components.v1 as components

st.set_page_config(layout="wide", page_title="VISHAYAMITRA", page_icon="🧠")
st.sidebar.page_link("app.py", label="Home", icon="🏠")
st.sidebar.page_link("pages/home.py", label="ChatBI", icon="💬")
st.sidebar.page_link("pages/pattern.py", label="Pattern Identifier", icon="📈")
st.sidebar.page_link("pages/visualization.py", label="Data Visualizer", icon="✨")
st.sidebar.page_link("pages/sqldata.py", label="Database Connector", icon="💽")
st.sidebar.page_link("pages/data_editor.py", label="Database Editor", icon="✏️")




os.environ["PANDASAI_API_KEY"] = st.secrets['PANDASAI_API_KEY']
os.environ["GOOGLE_API_KEY"] = st.secrets['GOOGLE_API_KEY']
llm = ChatGoogleGenerativeAI(model="gemini-pro")


class StreamlitResponse(ResponseParser):
    
    def __init__(self, context) -> None:
        super().__init__(context)

    def format_dataframe(self, result):
        st.dataframe(result["value"])
        return

    def format_plot(self, result):
        st.image(result["value"])
        return

    def format_other(self, result):
        st.write(result['value'])
        return

st.title(":orange[VISHAYAMITRA]")
uploaded_file = st.file_uploader("Choose Your Data file", type=["csv","xlsx","json"])




if uploaded_file is not None:
    st.session_state['data'] = pd.DataFrame()
    if uploaded_file.name.split(".")[-1]=='json':
        st.session_state['data'] = pd.read_json(uploaded_file)
    elif uploaded_file.name.split(".")[-1]=='csv':
        st.session_state['data'] = pd.read_csv(uploaded_file)
    elif uploaded_file.name.split(".")[-1]=='xlsx':
        st.session_state['data']= pd.read_excel(uploaded_file)
    
    st.write(st.session_state['data'].head())
    s = Agent(st.session_state['data'], config={'response_parser':StreamlitResponse})
    agent = create_pandas_dataframe_agent(
    st.session_state['data'],
    verbose=True
)
    
    with st.expander("Data Profile Report"):
        st.write("Sorry Cannot process the data")
    prompt = st.text_area("Ask a question")
    if st.button("Submit"):
        if prompt:
            result=""
            with st.spinner(text="In progress..."):
                result=s.chat(prompt)
            st.write(result)
        else:
            st.warning("Please enter a question")
