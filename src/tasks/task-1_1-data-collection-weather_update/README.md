# Update weather data from new source: open-meteo.com 

Based on the analysis on our first weather data [....] by Daniel Opanubi we came to the conclusion that this dataset is basically useless. It contains too many missing values on important factors like precipitation, snowfall and windspeed. 

Using the historical weather API from https://open-meteo.com/ I gathered more complete data again. Performing EDA on the dataset shows there are no missing values in this dataset. 

# data
- Based on the size of the data It has been decided to use daily data instead of hourly
- daily data means that the 24 h average is given for all the datapoints
- These are the columns that have been gathered from the API: 

| Column  | Description                                             | Unit                                    
|---------|-------------------------------------------------------------------------------------
| city | city in Brussel based on latitude and longitude in API     | name   
| time    | The datetime of the observation                         | datetime                              
| weathercode    | The most severe weather condition on a given day | WMO code                                                           
| temperature_min   |   Maximum and minimum daily air temperature at 2 meters above ground   | °C (°F)          
| temperature_max   |   Maximum and minimum daily air temperature at 2 meters above ground   | °C (°F)  
|precip   |  	Sum of daily precipitation (including rain, showers and snowfall)       | mm                                            
| rain  | Sum of daily rain   | mm                                          
| snowfall    | Sum of daily snowfall                     | cm                                             
| precip_h   | The number of hours with rain)    | hours                                            
| wind_speed   | Maximum wind speed and gusts on a day            | km/h (mph, m/s, knots)                                              
| wind_gusts   | Maximum wind speed and gusts on a day            | km/h (mph, m/s, knots)    
| wind_dir   | Dominant wind direction               | °           
| information_retrieved_at  | date and time of information retrieval    | datetime                                      
                                               

# Files in this folder
1. I uploaded the CSV file of the final weather dataset to our joint google drive here: https://drive.google.com/drive/folders/1GwxT3vZ10zS9CAdR--LfAJqBXOJVWUpX
2. I share the notebook that is used to get the data from the API and store it into a pandas dataframe 
