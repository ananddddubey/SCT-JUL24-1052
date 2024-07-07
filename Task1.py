import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = "API_SP.POP.TOTL_DS2_en_csv_v2_350067.csv"
df = pd.read_csv(file_path, skiprows=4)  

df_melted = df.melt(id_vars=['Country Name', 'Country Code'], 
                    var_name='Year', 
                    value_name='Population')

df_melted['Year'] = pd.to_numeric(df_melted['Year'], errors='coerce')
df_melted = df_melted.dropna(subset=['Year', 'Population'])

df_filtered = df_melted[(df_melted['Year'] >= 2014) & (df_melted['Year'] <= 2023)]

df_avg = df_filtered.groupby('Country Name')['Population'].mean().sort_values(ascending=False).reset_index()

plt.figure(figsize=(15, 10))
sns.barplot(x='Country Name', y='Population', data=df_avg.head(20))
plt.title('Top 20 Countries by Average Population (2014-2023)', fontsize=16)
plt.xlabel('Country', fontsize=12)
plt.ylabel('Average Population', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.ticklabel_format(style='plain', axis='y')

plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 10))
sns.histplot(df_avg['Population'], kde=True, bins=30)
plt.title('Distribution of Average Population Across Countries (2014-2023)', fontsize=16)
plt.xlabel('Average Population', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.ticklabel_format(style='plain', axis='x')

plt.tight_layout()
plt.show()
