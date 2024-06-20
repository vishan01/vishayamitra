import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

st.sidebar.page_link("app.py", label="Home", icon="🏠")
st.sidebar.page_link("pages/home.py", label="ChatBI", icon="💬")
st.sidebar.page_link("pages/pattern.py", label="Pattern Identifier", icon="📈")
st.sidebar.page_link("pages/visualization.py", label="Data Visualizer", icon="✨")
st.sidebar.page_link("pages/sqldata.py", label="Database Connector", icon="💽")



def train_linear_regression(x_train, y_train):
    model = LinearRegression()
    model.fit(x_train, y_train)
    return model.predict

def train_decision_tree(x_train, y_train):
    model = DecisionTreeRegressor()
    model.fit(x_train, y_train)
    return model.predict

def train_random_forest(x_train, y_train):
    model = RandomForestRegressor()
    model.fit(x_train, y_train)
    return model.predict

# Example usage
x_train = np.array([[1], [2], [3], [4], [5]])
y_train = np.array([2, 4, 6, 8, 10])

with st.container():
    model_type = st.selectbox("Select Model",index=None, placeholder="Select Model" ,options=["Linear Regression", "Decision Tree", "Random Forest"])
    x_input = st.number_input("Enter x value", value=0)
    try:
        if model_type == "Linear Regression":
            y_pred = train_linear_regression(x_train, y_train)([[x_input]])
        elif model_type == "Decision Tree":
            y_pred = train_decision_tree(x_train, y_train)([[x_input]])
        elif model_type == "Random Forest":
            y_pred = train_random_forest(x_train, y_train)([[x_input]])
        
        st.write("Predicted y value:", y_pred)

    except:
        st.write("Please select different model or enter different x value.")