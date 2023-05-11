import streamlit as st
from PIL import Image
import pickle
import os
from utils import DAY_OF_ARRIVAL_DICT, FINAL_DESTINATION_DICT, LINE_NO_DICT, RELATION_DICT, START_LOCATION_DICT

#import xgboost as xgb
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
#import plotly.express as px
#from plotly.subplots import make_subplots
#import plotly.graph_objects as go
#import calendar

#%matplotlib inline
import math
import urllib


#scikitlearn
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import accuracy_score

# tensorflow
# import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
# from tensorflow.keras import layers
#import tensorflow_decision_forests as tfdf

header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()
model_Prediction = st.container()
my_dirs = os.listdir()
app_path = os.path.abspath(os.curdir)

if '.streamlit' in my_dirs:
    images_path = f'{app_path}/src/results/Prediction_app/data_for_streamlitApp'
    model_path = f'{app_path}/src/results/Prediction_app/ml_model'
else:
    images_path = f'{app_path}/data_for_streamlitApp'
    model_path = f'{app_path}/ml_model'


with header:
    st.title('Welcome to our Train Prediction Project')
    st.text('In this project, we built models on how to predict train delays in Belgium')

with dataset:
    st.header('Belgium Train dataset')
    st.text('The dataset used in this project is the Train delay dataset for Belgium Below is the first 5 columns of the dataset')
   

#     belgium_data = pd.read_parquet('data_for_streamlitApp/test.parquet')
#     st.write(belgium_data.head())


    st.header('Here are a few insights we gathered from our data analysis')


image1 = Image.open(f'{images_path}/most_busy_train_day.png')
image5 = Image.open(f'{images_path}/days_of_arrival.png')
image2 = Image.open(f'{images_path}/delayed_departure.png')
image3 = Image.open(f'{images_path}/delayed_arrival.png')
image4 = Image.open(f'{images_path}/delayed_arrival_hours.png')


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


    
#Loading up the model we created
    
    


# model = Sequential([
#     Dense(256, activation='relu'), #64
#     Dense(256, activation='relu'), #64
#     Dense(128, activation='relu'), #64
#     Dense(128, activation='relu'),#32
#     Dense(1)
# ])

# #model.load_model('OmdenaTrainDelaysTensorFlow1.pkl')
# loaded_model = pickle.load(open('OmdenaTrainDelaysTensorFlow1.pkl', 'rb'))

#Caching the model for faster loading
@st.cache

#model = pickle.load(open('OmdenaTrainDelaysTensorFlow','rb'))

#st.markdown("Here we are using Train Number as the input to predict the Train Delay")


# Define the prediction function

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

model = keras.models.load_model(model_path, 
                        custom_objects={'rmse':rmse})



def predict(line_no_arr, relation, hour_of_arrival, day_of_arrival, start_location, final_stop_location):
    LINE_NO_ARR = LINE_NO_DICT[line_no_arr]
    RELATION = RELATION_DICT[relation]
    DAY_OF_ARRIVAL = DAY_OF_ARRIVAL_DICT[day_of_arrival]
    START_LOCATION = START_LOCATION_DICT[start_location]
    FINAL_STOP_LOCATION = FINAL_DESTINATION_DICT[final_stop_location]
    HOUR_OF_ARRIVAL = int(hour_of_arrival)

    data = dict(zip(
        ['LINE_NO_ARR', 'RELATION', 'HOUR_OF_ARRIVAL', 'DAY_OF_ARRIVAL', 'START_LOCATION', 'FINAL_STOP_LOCATION'],
        [LINE_NO_ARR, RELATION, HOUR_OF_ARRIVAL, DAY_OF_ARRIVAL, START_LOCATION, FINAL_STOP_LOCATION]

    ))

    input_df = pd.DataFrame(data, index=[0])
    print(input_df.dtypes)
    prediction = np.ravel(model.predict(input_df))[0]

    return prediction


st.title("Train Delay Predictions")

st.header('Please fill this form below')
RELATION = st.selectbox('Relation', list(RELATION_DICT.keys()))


LINE_NO_ARR = st.selectbox('Line Number', list(LINE_NO_DICT.keys()))


HOUR_OF_ARRIVAL = st.selectbox('Hour of Arrival:', ['20', '21',  '6',  '7', '22', '14', '15',  '8',  '9', '17', '18', '19', '23', '16', '11', '12', '13',
       '10',  '5',  '0',  '4',  '1',  '2'])

DAY_OF_ARRIVAL = st.selectbox('Day of Arrival', list(DAY_OF_ARRIVAL_DICT.keys()))
START_LOCATION = st.selectbox('Start Location', list(START_LOCATION_DICT.keys()))
FINAL_STOP_LOCATION = st.selectbox('Final Location', list(FINAL_DESTINATION_DICT.keys()))




if st.button('Predict Delay'):
    seconds = predict(LINE_NO_ARR, RELATION, HOUR_OF_ARRIVAL, DAY_OF_ARRIVAL, START_LOCATION, FINAL_STOP_LOCATION)
    st.success(f'The predicted seconds of delay is {seconds}')
    # st.write(seconds)


