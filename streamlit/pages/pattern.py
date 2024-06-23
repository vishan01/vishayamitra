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
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from statsmodels.tsa.arima.model import ARIMA


st.set_page_config(layout="wide", page_title="VISHAYAMITRA", page_icon="🧠")

st.sidebar.page_link("app.py", label="Home", icon="🏠")
st.sidebar.page_link("pages/home.py", label="ChatBI", icon="💬")
st.sidebar.page_link("pages/pattern.py", label="Pattern Identifier", icon="📈")
st.sidebar.page_link("pages/visualization.py", label="Data Visualizer", icon="✨")
st.sidebar.page_link("pages/sqldata.py", label="Database Connector", icon="💽")


class Regression:
    def train_linear_regression(self, x_train, y_train):
        model = LinearRegression()
        model.fit(x_train, y_train)
        return model
    
    def train_decision_tree(self, x_train, y_train):
        model = DecisionTreeRegressor()
        model.fit(x_train, y_train)
        return model
    
    def train_random_forest(self, x_train, y_train):
        model = RandomForestRegressor()
        model.fit(x_train, y_train)
        return model
    
    def train_support_vector_regression(self, x_train, y_train):
        model = SVR()
        model.fit(x_train, y_train)
        return model
    
    def train_k_neighbors(self, x_train, y_train):
        model = KNeighborsRegressor()
        model.fit(x_train, y_train)
        return model
    
    def train_adaboost(self, x_train, y_train):
        model = AdaBoostRegressor()
        model.fit(x_train, y_train)
        return model
    
    def train_neural_network(self, x_train, y_train):
        model = MLPRegressor()
        model.fit(x_train, y_train)
        return model
    
    def train_gaussian_process(self, x_train, y_train):
        model = GaussianProcessRegressor()
        model.fit(x_train, y_train)
        return model



class Classification:
    def train_logistic_regression(self, x_train, y_train):
        model = LogisticRegression()
        model.fit(x_train, y_train)
        return model
    
    def train_decision_tree(self, x_train, y_train):
        model = DecisionTreeClassifier()
        model.fit(x_train, y_train)
        return model
    
    def train_random_forest(self, x_train, y_train):
        model = RandomForestClassifier()
        model.fit(x_train, y_train)
        return model
    
    def train_support_vector_machine(self, x_train, y_train):
        model = SVC()
        model.fit(x_train, y_train)
        return model
    
    def train_k_neighbors(self, x_train, y_train):
        model = KNeighborsClassifier()
        model.fit(x_train, y_train)
        return model
    
    def train_adaboost(self, x_train, y_train):
        model = AdaBoostClassifier()
        model.fit(x_train, y_train)
        return model
    
    def train_neural_network(self, x_train, y_train):
        model = MLPClassifier()
        model.fit(x_train, y_train)
        return model
    
    def train_gaussian_process(self, x_train, y_train):
        model = GaussianProcessClassifier()
        model.fit(x_train, y_train)
        return model


class AnomalyDetection:
    def train_isolation_forest(self, x_train):
        model = IsolationForest()
        model.fit(x_train)
        return model
    
    def train_local_outlier_factor(self, x_train):
        model = LocalOutlierFactor()
        model.fit(x_train)
        return model
    
    def train_one_class_svm(self, x_train):
        model = OneClassSVM()
        model.fit(x_train)
        return model


class Clustering:
    def train_k_means(self, x_train):
        model = KMeans()
        model.fit(x_train)
        return model
    
    def train_agglomerative(self, x_train):
        model = AgglomerativeClustering()
        model.fit(x_train)
        return model
    
    def train_dbscan(self, x_train):
        model = DBSCAN()
        model.fit(x_train)
        return model


class TimeSeries:
    def train_arima(self, x_train):
        model = ARIMA()
        model.fit(x_train)
        return model
    




st.title(":Orange[Vishayamitra] Pattern Identifier")
# Example usage
uploaded_file = st.file_uploader("Choose a file", type=["csv"])
if uploaded_file!=None:
    data=pd.read_csv(uploaded_file)
    st.write(data.head())
    task_type = st.selectbox("Select Task Type", index=None, options=["Regression", "Classification", "Anomaly Detection", "Clustering", "Time Series"])
    try:
        if task_type == "Regression":
            x = st.selectbox("Select x value", data.columns)
            y = st.selectbox("Select y value", data.columns)
            x_train = data[x].values.reshape(-1, 1)
            y_train = data[y].values
            model_type = st.selectbox("Select Model", index=None, options=["Linear Regression", "Decision Tree", "Random Forest", "Support Vector Regression", "K Neighbors", "AdaBoost", "Neural Network", "Gaussian Process"])
            if model_type == "Linear Regression":
                model = Regression().train_linear_regression(x_train, y_train)
            elif model_type == "Decision Tree":
                model = Regression().train_decision_tree(x_train, y_train)
            elif model_type == "Random Forest":
                model = Regression().train_random_forest(x_train, y_train)
            elif model_type == "Support Vector Regression":
                model = Regression().train_support_vector_regression(x_train, y_train)
            elif model_type == "K Neighbors":
                model = Regression().train_k_neighbors(x_train, y_train)
            elif model_type == "AdaBoost":
                model = Regression().train_adaboost(x_train, y_train)
            elif model_type == "Neural Network":
                model = Regression().train_neural_network(x_train, y_train)
            elif model_type == "Gaussian Process":
                model = Regression().train_gaussian_process(x_train, y_train)
            
            x_input = st.text_input("Enter x value")

            if st.button("Predict"):
                try:
                    st.write("Model:", model_type)
                    st.write("accuracy Score:", model.score(x_train, y_train))
                    x_input = np.array(x_input).reshape(1, -1)
                    y_pred = model.predict(x_input)
                    st.write("Predicted y value:", y_pred)
                except:
                    st.write("Please select different model or enter different x value.")
        
        elif task_type == "Classification":
            x = st.selectbox("Select x value", data.columns)
            y = st.selectbox("Select y value", data.columns)
            x_train = data[x].values.reshape(-1, 1)
            y_train = data[y].values

            model_type = st.selectbox("Select Model", index=None, options=["Logistic Regression", "Decision Tree", "Random Forest", "Support Vector Machine", "K Neighbors", "AdaBoost", "Neural Network", "Gaussian Process"])
            if model_type == "Logistic Regression":
                model = Classification().train_logistic_regression(x_train, y_train)
            elif model_type == "Decision Tree":
                model = Classification().train_decision_tree(x_train, y_train)
            elif model_type == "Random Forest":
                model = Classification().train_random_forest(x_train, y_train)
            elif model_type == "Support Vector Machine":
                model = Classification().train_support_vector_machine(x_train, y_train)
            elif model_type == "K Neighbors":
                model = Classification().train_k_neighbors(x_train, y_train)
            elif model_type == "AdaBoost":
                model = Classification().train_adaboost(x_train, y_train)
            elif model_type == "Neural Network":
                model = Classification().train_neural_network(x_train, y_train)
            elif model_type == "Gaussian Process":
                model = Classification().train_gaussian_process(x_train, y_train)
            
            x_input = st.text_input("Enter x value")

            if st.button("Predict"):
                try:
                    st.write("Model:", model_type)
                    st.write("accuracy Score:", model.score(x_train, y_train))
                    x_input = np.array(x_input).reshape(1, -1)
                    y_pred = model.predict(x_input)
                    st.write("Predicted y value:", y_pred)
                except:
                    st.write("Please select different model or enter different x value.")
        
        elif task_type == "Anomaly Detection":
            x = st.selectbox("Select x value", data.columns)
            x_train = data[x].values.reshape(-1, 1)
            model_type = st.selectbox("Select Model", index=None, options=["Isolation Forest", "Local Outlier Factor", "One Class SVM"])
            if model_type == "Isolation Forest":
                model = AnomalyDetection().train_isolation_forest(x_train)
            elif model_type == "Local Outlier Factor":
                model = AnomalyDetection().train_local_outlier_factor(x_train)
            elif model_type == "One Class SVM":
                model = AnomalyDetection().train_one_class_svm(x_train)
            
            if st.button("Predict"):
                try:
                    st.write("Model:", model_type)
                    y_pred = model.predict(x_train)
                    data["result"]=y_pred
                    x.append("result")
                    st.write("Predicted value")
                    st.dataframe(data[x])
                except:
                    st.write("Please select different model or enter different x value.")
        
        elif task_type == "Clustering":
            x = st.selectbox("Select x value", data.columns)
            x_train = data[x].values.reshape(-1, 1)
            model_type = st.selectbox("Select Model", index=None, options=["K Means", "Agglomerative", "DBSCAN"])
            if model_type == "K Means":
                model = Clustering().train_k_means(x_train)
            elif model_type == "Agglomerative":
                model = Clustering().train_agglomerative(x_train)
            elif model_type == "DBSCAN":
                model = Clustering().train_dbscan(x_train)
            
            if st.button("Predict"):
                try:
                    st.write("Model:", model_type)
                    y_pred = model.predict(x_train)
                    data["result"]=y_pred
                    x.append("result")
                    st.write("Predicted value")
                    st.dataframe(data[x])
                except:
                    st.write("Please select different model or enter different x value.")
        
        elif task_type == "Time Series":
            x = st.selectbox("Select x value", data.columns)
            x_train = data[x].values
            model_type = st.selectbox("Select Model", index=None, options=["ARIMA"])
            if model_type == "ARIMA":
                model = TimeSeries().train_arima(x_train)
            
            if st.button("Predict"):
                try:
                    st.write("Model:", model_type)
                    y_pred = model.predict(x_train)
                    data["result"]=y_pred
                    x.append("result")
                    st.write("Predicted value")
                    st.dataframe(data[x])
                except:
                    st.write("Please select different model or enter different x value.")
    except:
        st.write("Please select different task type or model or preprocess the data before training the model.")