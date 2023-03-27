# Update weather data from new source: open-meteo.com 

Based on the analysis on our first weather data [....] by Daniel Opanubi we came to the conclusion that this dataset is basically useless. It contains too many missing values on important factors like precipitation, snowfall and windspeed. 

Using the historical weather API from https://open-meteo.com/ I gathered more complete data again. Performing EDA on the dataset shows there are no missing values in this dataset. 

# Files in this folder
1. I pushed the CSV file of the final weather dataset to github
2. I share the notebook that is used to get the data from the API and store it into a pandas dataframe 
