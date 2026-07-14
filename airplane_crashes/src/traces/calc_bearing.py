
import numpy as np

# Soruce: https://www.movable-type.co.uk/scripts/latlong.html
def calc_bearing(df_icao):
    lat1 = np.radians(df_icao['lat'])
    lat2 = np.radians(df_icao['lat'].shift(-1))
    dlon = np.radians(df_icao['lon'].shift(-1) - df_icao['lon'])
    
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    
    bearing = np.degrees(np.arctan2(x, y)) % 360
    return bearing

