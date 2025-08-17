
import requests
import pandas as pd

def get_nasa_power_data(lat, lon, start_date, end_date, parameters):
    base_url = "https://power.larc.nasa.gov/api/temporal/daily/point?parameters={}&community=AG&longitude={}&latitude={}&start={}&end={}&format=JSON"
    params_str = ",".join(parameters)
    api_url = base_url.format(params_str, lon, lat, start_date, end_date)
    
    response = requests.get(api_url)
    data = response.json()
    
    if 'properties' not in data:
        print("Error: No data found or invalid response.")
        print(data)
        return None
        
    df = pd.DataFrame(data['properties']['parameter'])
    df.index = pd.to_datetime(df.index, format='%Y%m%d')
    df.index.name = 'Date'
    return df

if __name__ == "__main__":
    lat = 5.55  # Latitude for Accra, Ghana
    lon = -0.22 # Longitude for Accra, Ghana
    start_date = "20190101"
    end_date = "20241231"
    parameters = ["T2M", "RH2M", "PRECTOTCORR"]
    
    df_env = get_nasa_power_data(lat, lon, start_date, end_date, parameters)
    
    if df_env is not None:
        df_env.columns = ['Temperature_2m', 'Relative_Humidity_2m', 'Precipitation']
        df_env.to_csv("accra_environmental_data.csv")
        print("Environmental data saved to accra_environmental_data.csv")
    else:
        print("Failed to retrieve environmental data.")


