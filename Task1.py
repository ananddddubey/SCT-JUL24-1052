import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = "API_SP.POP.TOTL_DS2_en_csv_v2_350067.csv"
df = pd.read_csv(file_path, skiprows=4)  # Skip the first 4 rows as they usually contain metadata

# Melt the dataframe to convert years from columns to rows
df_melted = df.melt(id_vars=['Country Name', 'Country Code'], 
                    var_name='Year', 
                    value_name='Population')

# Convert Year to numeric and drop any non-year columns
df_melted['Year'] = pd.to_numeric(df_melted['Year'], errors='coerce')
df_melted = df_melted.dropna(subset=['Year', 'Population'])

# Filter for the years 2014-2023
df_filtered = df_melted[(df_melted['Year'] >= 2014) & (df_melted['Year'] <= 2023)]

# Calculate average population for each country over 2014-2023
df_avg = df_filtered.groupby('Country Name')['Population'].mean().sort_values(ascending=False).reset_index()

# Create a bar chart of top 20 countries by average population
plt.figure(figsize=(15, 10))
sns.barplot(x='Country Name', y='Population', data=df_avg.head(20))
plt.title('Top 20 Countries by Average Population (2014-2023)', fontsize=16)
plt.xlabel('Country', fontsize=12)
plt.ylabel('Average Population', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.ticklabel_format(style='plain', axis='y')

# Adjust layout and display the chart
plt.tight_layout()
plt.show()

# Create a histogram of population distribution
plt.figure(figsize=(15, 10))
sns.histplot(df_avg['Population'], kde=True, bins=30)
plt.title('Distribution of Average Population Across Countries (2014-2023)', fontsize=16)
plt.xlabel('Average Population', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.ticklabel_format(style='plain', axis='x')

# Adjust layout and display the chart
plt.tight_layout()
plt.show()
