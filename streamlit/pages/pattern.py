import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor

st.set_page_config(layout="wide", page_title="VISHAYAMITRA", page_icon="🧠")

st.sidebar.page_link("app.py", label="Home", icon="🏠")
st.sidebar.page_link("pages/home.py", label="ChatBI", icon="💬")
st.sidebar.page_link("pages/pattern.py", label="Pattern Identifier", icon="📈")
st.sidebar.page_link("pages/visualization.py", label="Data Visualizer", icon="✨")
st.sidebar.page_link("pages/sqldata.py", label="Database Connector", icon="💽")




def train_support_vector_regression(x_train, y_train):
    model = SVR()
    model.fit(x_train, y_train)
    return model

def train_k_neighbors(x_train, y_train):
    model = KNeighborsRegressor()
    model.fit(x_train, y_train)
    return model

def train_adaboost(x_train, y_train):
    model = AdaBoostRegressor()
    model.fit(x_train, y_train)
    return model

def train_neural_network(x_train, y_train):
    model = MLPRegressor()
    model.fit(x_train, y_train)
    return model

def train_gaussian_process(x_train, y_train):
    model = GaussianProcessRegressor()
    model.fit(x_train, y_train)
    return model

def train_linear_regression(x_train, y_train):
    model = LinearRegression()
    model.fit(x_train, y_train)
    return model

def train_decision_tree(x_train, y_train):
    model = DecisionTreeRegressor()
    model.fit(x_train, y_train)
    return model

def train_random_forest(x_train, y_train):
    model = RandomForestRegressor()
    model.fit(x_train, y_train)
    return model


st.title(":Orange[Vishayamitra] Pattern Identifier")
# Example usage
uploaded_file = st.file_uploader("Choose a file", type=["csv"])
if uploaded_file!=None:
    data=pd.read_csv(uploaded_file)
    st.write(data.head())
    with st.container():
        x_features=st.multiselect("Select x features",options=data.columns)
        y_features=st.selectbox("Select y feature",index=None,options=data.columns)
        if x_features!=None and y_features!=None:
            x_train = data[x_features].values.reshape(-1,len(x_features))
            y_train = data[y_features].values
            model_type =st.selectbox("Select Model",index=None,options=["Linear Regression", "Decision Tree", "Random Forest", "Support Vector Regression", "K Neighbors", "AdaBoost", "Neural Network", "Gaussian Process"])
            x_input=[]
            for i in x_features:
                temp = st.number_input(f"Enter {i} value")
                x_input.append(temp)
            
            if st.button("Predict"):
                try:
                    if model_type == "Linear Regression":
                        model = train_linear_regression(x_train, y_train)
                    elif model_type == "Decision Tree":
                        model = train_decision_tree(x_train, y_train)
                    elif model_type == "Random Forest":
                        model = train_random_forest(x_train, y_train)
                    elif model_type == "Support Vector Regression":
                        model = train_support_vector_regression(x_train, y_train)
                    elif model_type == "K Neighbors":
                        model = train_k_neighbors(x_train, y_train)
                    elif model_type == "AdaBoost":
                        model = train_adaboost(x_train, y_train)
                    elif model_type == "Neural Network":
                        model = train_neural_network(x_train, y_train)
                    elif model_type == "Gaussian Process":
                        model = train_gaussian_process(x_train, y_train)
                    st.write("Model:", model_type)
                    st.write("accuracy Score:",model.score(x_train, y_train))
                    x_input=np.array(x_input).reshape(1,-1)
                    y_pred=model.predict(x_input)
                    st.write("Predicted y value:", y_pred)

                except:
                    st.write("Please select different model or enter different x value.")