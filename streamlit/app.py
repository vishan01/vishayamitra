
import os
import pandas as pd
from pandasai import Agent
from pandasai.responses.response_parser import ResponseParser
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent





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

st.title("Vishayamitra Prototype")
uploaded_file = st.file_uploader("Choose Your Data file", type=["csv","xlsx","json"])

if uploaded_file is not None:
    df = pd.DataFrame()
    if uploaded_file.name.split(".")[-1]=='json':
        df = pd.read_json(uploaded_file)
    elif uploaded_file.name.split(".")[-1]=='csv':
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.split(".")[-1]=='xlsx':
        df= pd.read_excel(uploaded_file)
    
    st.write(df.head())
    s = Agent(df, config={'response_parser':StreamlitResponse})
    agent = create_pandas_dataframe_agent(
    llm,
    df,
    verbose=True
)

    prompt = st.text_area("Ask a question")
    if st.button("Submit"):
        if prompt:
            
            with st.spinner(text="In progress..."):
                if "plot" in prompt:
                    response=s.chat(prompt)
                elif "data" in prompt:
                    response=s.chat(prompt)
                else:
                    response=agent.invoke(prompt)
                    response=response['output']
            st.write(response)
            
        else:
            st.warning("Please enter a question")