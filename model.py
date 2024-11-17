import requests
import pandas as pd
from geopy.distance import geodesic

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
params = {
    "format": "geojson",
    "starttime": "2022-01-01", 
    "endtime": "2024-11-09",  
    "minlatitude": 6.0,
    "maxlatitude": 36.0,
    "minlongitude": 68.0,
    "maxlongitude": 97.0,
    "minmagnitude": 1.0
}

response = requests.get(url, params=params)
data = response.json()


earthquake_data = pd.DataFrame([
    {
        'magnitude': eq['properties']['mag'],
        'depth': eq['geometry']['coordinates'][2], 
        'latitude': eq['geometry']['coordinates'][1],
        'longitude': eq['geometry']['coordinates'][0],
        'place': eq['properties']['place'],
        'time': eq['properties']['time']
    }
    for eq in data['features']
])

print(earthquake_data.head())

def calculate_min_distance_to_city(lat, lon, cities):

    distances = [geodesic((lat, lon), (city[1], city[2])).km for city in cities]
    return min(distances)

cities = [
    ('Delhi', 28.6139, 77.2090),
    ('Mumbai', 19.0760, 72.8777),
    ('Bangalore', 12.9716, 77.5946)
]
earthquake_data['distance_to_nearest_city'] = earthquake_data.apply(
    lambda row: calculate_min_distance_to_city(row['latitude'], row['longitude'], cities), axis=1
)
print(earthquake_data.head())

data = earthquake_data
def assign_severity(row):
    if row['magnitude'] >= 6.5 or (row['magnitude'] >= 5.5 and row['distance_to_nearest_city'] <= 50):
        return 'High'
    elif row['magnitude'] >= 4.5 or (row['magnitude'] >= 4.0 and row['distance_to_nearest_city'] <= 100):
        return 'Medium'
    else:
        return 'Low'
data['severity'] = data.apply(assign_severity, axis=1)

print(data)



X = data[['magnitude', 'depth', 'distance_to_nearest_city']]
y = data['severity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

def predict_manual_data():
    print("Enter test data for earthquake severity prediction:")
    magnitude = float(input("Magnitude: "))
    depth = float(input("Depth (in km): "))
    distance_to_nearest_city = float(input("Distance to nearest city (in km): "))
    manual_data = pd.DataFrame([[magnitude, depth, distance_to_nearest_city]], 
                               columns=['magnitude', 'depth', 'distance_to_nearest_city'])
    

    manual_data = scaler.transform(manual_data)
    
    
    prediction = model.predict(manual_data)
    print("\nPredicted Severity Level:", prediction[0])
    predict_manual_data()


predict_manual_data()
