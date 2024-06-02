
import os
import pandas as pd
from pandasai import SmartDataframe
from pandasai.responses.response_parser import ResponseParser
import streamlit as st


os.environ["PANDASAI_API_KEY"] = st.secrets['PANDASAI_API_KEY']

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
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df.head())
    s = SmartDataframe(df, config={'response_parser':StreamlitResponse})

    prompt = st.text_area("Ask a question")
    if st.button("Submit"):
        if prompt:
            response=s.chat(prompt)
            st.write(response)
            
        else:
            st.warning("Please enter a question")