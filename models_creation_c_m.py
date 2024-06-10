import itertools
import numpy as np
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import os
import pickle
import shutil
import zipfile
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

model_dir = os.path.join(os.getcwd(),'models')

# Check if the directory exists
if os.path.isdir(model_dir):
    # Create a timestamp for the zip file name
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

    # Create a zip file with the current timestamp in its name
    zip_filename = f'models_{timestamp}.zip'
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(model_dir):
            for file in files:
                zipf.write(os.path.join(root, file), arcname=file)

    # Remove all subdirectories in the original directory
    for dir in os.listdir(model_dir):
        dir_path = os.path.join(model_dir, dir)
        if os.path.isdir(dir_path):
            shutil.rmtree(dir_path)

    # Check if there are any zip files older than 7 days and delete them
    latest_zip_file = None
    for file in os.listdir():
        if file.startswith('models_') and file.endswith('.zip'):
            file_timestamp = datetime.strptime(file[7:21], '%Y%m%d%H%M%S')
            if datetime.now() - file_timestamp > timedelta(days=7):
                os.remove(file)
            else:
                if latest_zip_file is None or file_timestamp > latest_zip_file[1]:
                    latest_zip_file = (file, file_timestamp)

    # Remove the latest created folder if it's not the current folder
    if latest_zip_file is not None:
        latest_folder_path = os.path.join(model_dir, latest_zip_file[0][7:-4])
        if os.path.isdir(latest_folder_path) and latest_folder_path != os.getcwd():
            shutil.rmtree(latest_folder_path)

os.makedirs(model_dir, exist_ok=True)
#Target_cols = ['Country', 'Market']
Target_cols = ['Country', 'Market']
# Load and clean data
data = pd.read_csv('Data_final.csv', parse_dates=['Date'], dayfirst=True)
cols = ['Account Region']
data.drop(cols, axis=1, inplace=True)

days_per_month = data.groupby([data['Date'].dt.year, data['Date'].dt.month])['Date'].nunique()
if days_per_month.iloc[-1] < 28:
    days_per_month = days_per_month.iloc[:-1]
mask = data['Date'].dt.strftime('%Y-%m').isin(days_per_month.index.map(lambda x: f'{x[0]}-{x[1]:02d}'))
data = data.loc[mask]
data.sort_values('Date', inplace=True)
data = data.dropna().rename(columns={'Sum of Count Of SR': 'Count Of SR','Account Region' : 'Region'})
data['z_score'] = np.abs((data['Count Of SR'] - data['Count Of SR'].mean()) / data['Count Of SR'].std())
data = data[data['z_score'] <= 3]

data['Country'].replace(['Canada','Mexico'],'Canada',inplace=True)
data['Country'].replace(['United States', 'Russia','Aruba','Bermuda','British Virgin Islands','Guyana',
                         'Cayman Islands','Haiti','Puerto Rico','The Bahamas','Turks and Caicos Islands'
                         ],'United States',inplace=True)

data.sort_values('Date', inplace=True)
data.dropna(inplace=True)
data.rename(columns={'Sum of Count Of SR': 'Count Of SR'}, inplace=True)

# Resample data by month
data = data.set_index('Date')

# Get unique countries and markets
countries = data['Country'].unique()
markets = data['Market'].unique()
# Iterate over each combination of country and market
for country in countries:
    for market in markets:
        # Filter data for the current country and market
        filtered_data = data[(data['Country'] == country) & (data['Market'] == market)]
        
        if filtered_data.shape[0] > 100:
            try:
              print(f'\nFor Country: {country}, Market: {market}, Below are model details & forecast')
              y = filtered_data['Count Of SR'].resample('MS').sum()
              print('Count Of SR details:\n', y.describe())
              # Define the range of p, d, and q values
              p = d = q = range(0, 2)
              # Generate all possible combinations of p, d, and q
              pdq = list(itertools.product(p, d, q))
              # Generate all possible combinations of seasonal p, d, and q
              seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

              # Initialize variables to hold the best AIC and the corresponding parameters
              best_aic = np.inf
              best_param = None
              best_param_seasonal = None

              # Loop over all possible combinations of parameters and seasonal parameters
              for param in pdq:
                  for param_seasonal in seasonal_pdq:
                      try:
                          # Fit a SARIMAX model with the current combination of parameters
                          mod = sm.tsa.statespace.SARIMAX(y,
                                                          order=param,
                                                          seasonal_order=param_seasonal,
                                                          enforce_stationarity=False,
                                                          enforce_invertibility=False)
                          results = mod.fit()

                          # Check if the current model has a lower AIC than the best model so far
                          if results.aic < best_aic:
                              best_aic = results.aic
                              best_param = param
                              best_param_seasonal = param_seasonal
                      except:
                          continue

              # Print the best parameters and corresponding AIC value
              print("Best param: ", best_param)
              print("Best seasonal param: ", best_param_seasonal)
              print("Best AIC: ", best_aic)

              mod = sm.tsa.statespace.SARIMAX(y,
                                              order=best_param,
                                              seasonal_order=best_param_seasonal,
                                              enforce_stationarity=False,
                                              enforce_invertibility=False)

              results = mod.fit()
              
              file_path = os.path.join(model_dir, country + "_" + market + ".pickle")
              # Save the model
              with open(file_path, 'wb') as f:
                  pickle.dump(results, f)

            except Exception as e:
              print('Exception in model building')

        else:
            print(f'Not enough data for given inputs')


