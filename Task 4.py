import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Load the data
df = pd.read_csv(r"C:\Users\jidub\Downloads\US_Accidents_March23.csv\US_Accidents_March23.csv")

# 1. Data overview and preprocessing
print("Data shape:", df.shape)
print("\nColumns:", df.columns)

# Convert Start_Time and End_Time to datetime
df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
df['End_Time'] = pd.to_datetime(df['End_Time'], errors='coerce')

# Extract hour, day of week, and month from Start_Time
df['Hour'] = df['Start_Time'].dt.hour
df['Day_of_Week'] = df['Start_Time'].dt.dayofweek
df['Month'] = df['Start_Time'].dt.month

# Convert boolean columns to proper boolean type
bool_columns = ['Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit', 'Railway', 'Roundabout', 'Station', 'Stop', 'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop']
for col in bool_columns:
    df[col] = df[col].map({'False': False, 'True': True})

# Handle missing values
df = df.fillna(method='ffill')

# 2. Temporal analysis
plt.figure(figsize=(15, 5))

plt.subplot(131)
df['Hour'].value_counts().sort_index().plot(kind='bar')
plt.title('Accidents by Hour')
plt.xlabel('Hour')
plt.ylabel('Number of Accidents')

plt.subplot(132)
df['Day_of_Week'].value_counts().sort_index().plot(kind='bar')
plt.title('Accidents by Day of Week')
plt.xlabel('Day of Week (0=Monday, 6=Sunday)')
plt.ylabel('Number of Accidents')

plt.subplot(133)
df['Month'].value_counts().sort_index().plot(kind='bar')
plt.title('Accidents by Month')
plt.xlabel('Month')
plt.ylabel('Number of Accidents')

plt.tight_layout()
plt.savefig('temporal_analysis.png')
plt.close()

# 3. Weather analysis
plt.figure(figsize=(12, 6))
df['Weather_Condition'].value_counts().nlargest(10).plot(kind='bar')
plt.title('Top 10 Weather Conditions in Accidents')
plt.xlabel('Weather Condition')
plt.ylabel('Number of Accidents')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('weather_conditions.png')
plt.close()

# Correlation between weather factors and severity
weather_cols = ['Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)']
correlation = df[weather_cols + ['Severity']].corr()['Severity'].sort_values(ascending=False)
print("\nCorrelation between weather factors and accident severity:")
print(correlation)

# 4. Road condition analysis
road_features = ['Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit', 'Railway', 'Roundabout', 'Station', 'Stop', 'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop']

plt.figure(figsize=(12, 6))
df[road_features].sum().sort_values(ascending=False).plot(kind='bar')
plt.title('Frequency of Road Features in Accidents')
plt.xlabel('Road Feature')
plt.ylabel('Number of Accidents')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('road_features.png')
plt.close()

# 5. Geospatial analysis
plt.figure(figsize=(15, 10))
plt.hexbin(df['Start_Lng'], df['Start_Lat'], gridsize=20, cmap='YlOrRd')
plt.colorbar(label='Number of Accidents')
plt.title('Heatmap of Accident Locations')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.tight_layout()
plt.savefig('geospatial_heatmap.png')
plt.close()

# 6. Severity analysis
plt.figure(figsize=(10, 5))
df['Severity'].value_counts().sort_index().plot(kind='bar')
plt.title('Distribution of Accident Severity')
plt.xlabel('Severity')
plt.ylabel('Number of Accidents')
plt.tight_layout()
plt.savefig('severity_distribution.png')
plt.close()

# Correlation between various factors and severity
factors = ['Distance(mi)', 'Temperature(F)', 'Wind_Speed(mph)', 'Visibility(mi)', 'Precipitation(in)']
correlation = df[factors + ['Severity']].corr()['Severity'].sort_values(ascending=False)
print("\nCorrelation between various factors and accident severity:")
print(correlation)

# 7. Visualization of key findings
top_factors = correlation.nlargest(5)

plt.figure(figsize=(12, 6))
top_factors.plot(kind='bar')
plt.title('Top 5 Factors Contributing to Accident Severity')
plt.xlabel('Factor')
plt.ylabel('Correlation with Severity')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('top_severity_factors.png')
plt.close()

print("\nAnalysis complete. Visualizations have been saved as PNG files.")
