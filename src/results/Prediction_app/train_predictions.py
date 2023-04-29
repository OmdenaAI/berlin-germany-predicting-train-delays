import streamlit as st
from PIL import Image
import pickle

#import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#import plotly.express as px
#from plotly.subplots import make_subplots
#import plotly.graph_objects as go
import calendar

#%matplotlib inline
import math
import urllib


#scikitlearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score

# tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras import layers
#import tensorflow_decision_forests as tfdf

header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()
model_Prediction = st.container()


with header:
    st.title('Welcome to our Train Prediction Project')
    st.text('In this project, we built models on how to predict train delays in Belgium')

with dataset:
    st.header('Belgium Train dataset')
    st.text('The dataset used in this project is the Train delay dataset for Belgium Below is the first 5 columns of the dataset')
   

#     belgium_data = pd.read_parquet('data_for_streamlitApp/test.parquet')
#     st.write(belgium_data.head())


    st.header('Here are a few insights we gathered from our data analysis')


image1 = Image.open('data_for_streamlitApp/most_busy_train_day.png')
image5 = Image.open('data_for_streamlitApp/days_of_arrival.png')
image2 = Image.open('data_for_streamlitApp/delayed_departure.png')
image3 = Image.open('data_for_streamlitApp/delayed_arrival.png')
image4 = Image.open('data_for_streamlitApp/delayed_arrival_hours.png')


# Comment: 
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        st.image(image1, caption='Most busy train day')
    with col2:
        st.image(image5, caption='Days of Arrival')

# Comment: 
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        st.image(image2, caption='Delayed Departure')
    with col2:
        st.image(image3, caption='Delayed Arrival')


# Comment: 
with st.container():
    st.image(image4, caption='delayed arrival hours') 


with features:
    st.header('The features that were created')
    st.markdown('The features used in training our model are The hour of Arrival, the number of seconds it was delayed or if it arrived earlier than scheduled. We also used the top 40 train lines and One-hot encoded it')

with st.container():
    st.header('Time to train the model')
    st.text('Here you get to choose the hyperparameters of the model and see how the  performance changes ')

    #Loading up the Regression model we created

rf_model = pickle.load(open('OmdenaTrainDelaysRandomForestRegressor.pkl','rb'))

st.markdown("Here we are using Train Number as the input to predict the Train Delay")


#Input
st.subheader("Enter Your Train Number")
Train_number = st.text_input('')

#Predictions
st.subheader("Predicted Delay")
st.code((rf_model.predict([[Train_number]])))



parameters = {'bootstrap': True,
              'min_samples_leaf': 3,
              'n_estimators': 100, 
              'min_samples_split': 10,
              'max_features': 'sqrt',
              'max_depth': 100,
              'max_leaf_nodes': None}

RF_model = RandomForestRegressor(**parameters)
#Caching the model for faster loading
#@st.cache




with model_Prediction:
    st.header('')






