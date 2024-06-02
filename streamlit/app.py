
import os
import pandas as pd
from pandasai import SmartDataframe
from pandasai.responses.streamlit_response import StreamlitResponse
import streamlit as st


os.environ["PANDASAI_API_KEY"] = st.secrets['PANDASAI_API_KEY']

st.title("Data Analysis with PandasAI")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df.head())
    s = SmartDataframe(df, config={"response_parser": StreamlitResponse})

    prompt = st.text_area("Ask a question")
    if st.button("Submit"):
        if prompt:
            s.chat(prompt)
        else:
            st.warning("Please enter a question")