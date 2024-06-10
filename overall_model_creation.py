import itertools
import numpy as np
import statsmodels.api as sm
import pandas as pd
# import matplotlib.pyplot as plt
import warnings
import pickle
warnings.filterwarnings("ignore")

# Load and clean data
data = pd.read_csv('Data_final.csv', parse_dates=['Date'], dayfirst=True)
cols = ['Account Region', 'Country', 'Market']
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

# data.dropna(inplace=True)
# data.rename(columns={'Sum of Count Of SR': 'Count Of SR'}, inplace=True)

# Resample data by month
data = data.set_index('Date')
# data_grouped = data.groupby(pd.Grouper(freq='M')).sum()
# print(data_grouped)

y = data['Count Of SR'].resample('MS').sum()
print(y.describe())
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

# Save the model
with open('model_overall.pkl', 'wb') as f:
    pickle.dump(results, f)
