# Update weather data from new source: open-meteo.com 

Based on the analysis on our first weather data [....] by Daniel Opanubi we came to the conclusion that this dataset is basically useless. It contains too many missing values on important factors like precipitation, snowfall and windspeed. 

Using the historical weather API from https://open-meteo.com/ I gathered more complete data again. Performing EDA on the dataset shows there are no missing values in this dataset. 

# data
- Based on the size of the data I decided to use daily data and not hourly
- daily data means that the 24 h average is given for all the datapoints
- These are the columns that have been gathered from the API: 


# Files in this folder
1. I pushed the CSV file of the final weather dataset to our joint google drive here: https://drive.google.com/drive/folders/1GwxT3vZ10zS9CAdR--LfAJqBXOJVWUpX
2. I share the notebook that is used to get the data from the API and store it into a pandas dataframe 
