import streamlit as st
import pygwalker as pyg
import pandas as pd
from pygwalker.api.streamlit import StreamlitRenderer, init_streamlit_comm

st.set_page_config(layout="wide", page_title="Vishayamitra Data Visualizer",page_icon="📊")


init_streamlit_comm()

st.sidebar.page_link("app.py", label="Home", icon="🏠")
st.sidebar.page_link("pages/home.py", label="ChatBI", icon="💬")
st.sidebar.page_link("pages/pattern.py", label="Pattern Identifier", icon="📈")
st.sidebar.page_link("pages/visualization.py", label="Data Visualizer", icon="✨")
st.sidebar.page_link("pages/sqldata.py", label="Database Connector", icon="💽")


@st.cache_resource
def get_pyg_renderer(df) -> "StreamlitRenderer":
    # If you want to use feature of saving chart config, set `spec_io_mode="rw"`
    return StreamlitRenderer(df, spec="./gw_config.json", spec_io_mode="rw")
 
st.title(":orange[Vishayamitra] Data Visualizer")

uploaded_file = st.file_uploader("Choose Your Data file", type=["csv","xlsx","json"])
if uploaded_file is not None:
    df = pd.DataFrame()
    if uploaded_file.name.split(".")[-1]=='json':
        df = pd.read_json(uploaded_file)
    elif uploaded_file.name.split(".")[-1]=='csv':
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.split(".")[-1]=='xlsx':
        df= pd.read_excel(uploaded_file)

    renderer = get_pyg_renderer(df)
    renderer.explorer()
