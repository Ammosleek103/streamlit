import streamlit as st
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd 

# Load Iris dataset and prepare data
iris_dataset = load_iris()
features_data = iris_dataset.data
target_data = iris_dataset.target 

# Initialize and train the model
model = RandomForestClassifier()
model.fit(features_data, target_data)

def get_user_input():
    sepal_length = st.sidebar.slider('Sepal Length (cm)', min_value=float(features_data[:,0].min()), max_value=float(features_data[:,0].max()))
    sepal_width = st.sidebar.slider('Sepal Width (cm)', min_value=float(features_data[:,1].min()), max_value=float(features_data[:,1].max()))
    petal_length = st.sidebar.slider('Petal Length (cm)', min_value=float(features_data[:,2].min()), max_value=float(features_data[:,2].max()))
    petal_width = st.sidebar.slider('Petal Width (cm)', min_value=float(features_data[:,3].min()), max_value=float(features_data[:,3].max()))

    # Create DataFrame with user input
    input_data = {
        'sepal_length': sepal_length,
        'sepal_width': sepal_width,
        'petal_length': petal_length,
        'petal_width': petal_width
    }
    input_df = pd.DataFrame(input_data, index=[0])
    return input_df

# Get user input and display it
user_input_df = get_user_input()
st.subheader('Input Features')
st.write(user_input_df)

# Make predictions and display results
prediction = model.predict(user_input_df)
prediction_probabilities = model.predict_proba(user_input_df)

st.subheader('Prediction')
st.write(iris_dataset.target_names[prediction][0])

st.subheader('Prediction Probabilities')
st.write(pd.DataFrame(prediction_probabilities, columns=iris_dataset.target_names))
