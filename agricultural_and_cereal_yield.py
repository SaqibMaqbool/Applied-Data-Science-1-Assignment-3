
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import numpy as np
from scipy.optimize import curve_fit

# Load the dataset
data_file_path = 'API19.csv'
data_df = pd.read_csv(data_file_path, skiprows=4)

# Function for exponential curve fitting
def exponential_fit(x, a, b, c):
    return a * np.exp(b * x) + c

# Filter data for Agricultural Land (sq. km) and Cereal Yield (kg per hectare)
agricultural_land_data = data_df[data_df['Indicator Code'] == 'AG.LND.AGRI.K2']
cereal_yield_data = data_df[data_df['Indicator Code'] == 'AG.YLD.CREL.KG']

# Select the most recent year available for these datasets - using 2021
year = '2021'
agricultural_land_latest = agricultural_land_data[['Country Name', year]].rename(columns={year: 'Agricultural Land (sq. km)'})
cereal_yield_latest = cereal_yield_data[['Country Name', year]].rename(columns={year: 'Cereal Yield (kg per hectare)'})

# Merging the datasets
merged_data = pd.merge(agricultural_land_latest, cereal_yield_latest, on='Country Name')

# Drop NaN values for clustering
merged_data_clean = merged_data.dropna()

# KMeans Clustering
kmeans = KMeans(n_clusters=3)
merged_data_clean['Cluster'] = kmeans.fit_predict(merged_data_clean[['Agricultural Land (sq. km)', 'Cereal Yield (kg per hectare)']])

# Plotting the results with clustering
plt.figure(figsize=(12, 7))
sns.scatterplot(data=merged_data_clean, x='Agricultural Land (sq. km)', y='Cereal Yield (kg per hectare)', hue='Cluster', palette='viridis')
plt.title('Agricultural Land vs Cereal Yield with KMeans Clustering')
plt.xlabel('Agricultural Land (sq. km)')
plt.ylabel('Cereal Yield (kg per hectare)')
plt.grid(True)
plt.show()

# Curve Fitting for Cereal Yield
x_data = np.arange(len(merged_data_clean))
y_data = merged_data_clean['Cereal Yield (kg per hectare)'].values

# Fit the data to the exponential curve
params, params_covariance = curve_fit(exponential_fit, x_data, y_data, p0=[1, 1, 1])

# Plotting the curve fit
plt.figure(figsize=(12, 7))
plt.scatter(x_data, y_data, label='Data')
plt.plot(x_data, exponential_fit(x_data, *params), label='Fitted function', color='red')
plt.title('Curve Fitting for Cereal Yield')
plt.xlabel('Data Points')
plt.ylabel('Cereal Yield (kg per hectare)')
plt.legend()
plt.grid(True)
plt.show()
