# Add weather data to our potential model
### Get historical weather data for our prediction of train delays

# Goal
Using the meteostat python library I was able to generate a dataset with historical hourly weather data for all Belgian cities in the timeframe between 2020-2023.


# Data Set 
The Data Set has been generated using the meteostat python library: 
[link to the documentation](https://dev.meteostat.net/python/)
Each hour is represented by a Pandas DataFrame row which provides the weather data recorded at that time. 
These are the different columns:
| Column  | Description                                                                         | Type       |
|---------|-------------------------------------------------------------------------------------|------------|
| station | The Meteostat ID of the weather station (only if query refers to multiple stations) | String     |
| time    | The datetime of the observation                                                     | Datetime64 |
| temp    | The air temperature in °C                                                           | Float64    |
| dwpt    | The dew point in °C                                                                 | Float64    |
| rhum    | The relative humidity in percent (%)                                                | Float64    |
| prcp    | The one hour precipitation total in mm                                              | Float64    |
| snow    | The snow depth in mm                                                                | Float64    |
| wdir    | The average wind direction in degrees (°)                                           | Float64    |
| wspd    | The average wind speed in km/h                                                      | Float64    |
| wpgt    | The peak wind gust in km/h                                                          | Float64    |
| pres    | The average sea-level air pressure in hPa                                           | Float64    |
| tsun    | The one hour sunshine total in minutes (m)                                          | Float64    |
| coco    | The weather condition code                                                          | Float64    |


## Files in this folder
- [The Code in which the dataset is being generated](berlin-germany-predicting-train-delays/src/tasks/task-1-data-collection-weather/weather_data.ipynb)

## To do:  
- normalize and interpolate time-series data 
- Transform dataset to better integrate it as features into the the train-delays dataset

