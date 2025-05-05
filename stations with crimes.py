import pandas as pd
from scipy.spatial import cKDTree
import folium
import webbrowser
import os


def convert_coordinates(coord_str):
    lat_str, lon_str = coord_str.split(',')
    lat_str = lat_str.replace('°', '').replace('N', '').replace('S', '').strip()
    lon_str = lon_str.replace('°', '').replace('E', '').replace('W', '').strip()

    latitude = float(lat_str)
    longitude = float(lon_str)

    if 'S' in coord_str:
        latitude = -latitude
    if 'W' in coord_str:
        longitude = -longitude

 
    return pd.Series([latitude, longitude], index=['latitude', 'longitude'])


def calculate_crime_percentage(station_lat, station_lon, crimes, radius_meters, seen_crimes):
   
    crime_coords = crimes[['Latitude', 'Longitude']].dropna().values 
    
    
    kdtree = cKDTree(crime_coords)

    
    radius_degrees = radius_meters / 111000.0
    
    
    station_coords = [station_lat, station_lon]  
    nearby_crimes = kdtree.query_ball_point(station_coords, radius_degrees)
    
   
    count_new_crimes = 0
    total_crimes = len(crime_coords)
    
    for crime_index in nearby_crimes:
        if crime_index not in seen_crimes:
            seen_crimes.add(crime_index)
            count_new_crimes += 1

   
    crime_percentage = (count_new_crimes / total_crimes) * 100 if total_crimes > 0 else 0
    
    return crime_percentage


stations = pd.read_csv('metropolitan_police_stations.csv')


stations[['latitude', 'longitude']] = stations['coordinates'].apply(convert_coordinates)


crimes = pd.read_csv('D:\\Program Files\\cbl\\data\\clean_data.csv')


radius_meters = 500  


total_crime_percentage_sum = 0
seen_crimes = set()


map_london = folium.Map(location=[51.5074, -0.1278], zoom_start=11)


for _, station in stations.iterrows():
    station_lat = station['latitude']
    station_lon = station['longitude']
    
   
    crime_percentage = calculate_crime_percentage(station_lat, station_lon, crimes, radius_meters, seen_crimes)
    
    
    print(f"Station: {station['Station']}, Crime percentage within {radius_meters} meters: {crime_percentage}%")
    
    
    total_crime_percentage_sum += crime_percentage
    
    
    folium.CircleMarker(
        location=[station_lat, station_lon],
        radius=5,
        color='black',
        fill=True,
        fill_color='black',
        popup=f"Station: {station['Station']}<br>Crime Percentage: {crime_percentage:.2f}%",
    ).add_to(map_london)


for _, row in crimes.iterrows():
    if pd.notnull(row['Latitude']) and pd.notnull(row['Longitude']):
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=3,
            color='red',
            fill=True,
            fill_color='red',
            fill_opacity=0.7
        ).add_to(map_london)


print(f"\nSum of Crime Percentages across all stations (without double-counting): {total_crime_percentage_sum}%")


map_london.save("stations_and_crimes_map_with_percentages.html")
webbrowser.open('file://' + os.path.realpath("stations_and_crimes_map_with_percentages.html"))
















