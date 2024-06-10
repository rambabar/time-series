import streamlit as st
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

st.set_option('deprecation.showPyplotGlobalUse', False)

# Create the Streamlit app
st.markdown("<h1 style='text-align: center;'> Forecasting App </h1>", unsafe_allow_html=True)
#st.title("Forecasting App")

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

model_dir = os.path.join(os.getcwd(), 'models')
pickle_files = os.listdir(model_dir)

# Get unique countries and markets from the list of file names
def get_country_and_market(filename):
    parts = filename.split('_')
    country = parts[0]
    market = parts[1]
    return country, market

countries = list(set([get_country_and_market(filename)[0] for filename in pickle_files]))
country_option = st.selectbox('Select Country', countries)

# Filter the pickle file names based on the selected country
filtered_files = [filename for filename in pickle_files if get_country_and_market(filename)[0] == country_option]

if filtered_files:
    markets = list(set([get_country_and_market(filename.split('.')[0])[1] for filename in filtered_files]))
    market_option = st.selectbox('Select Market', markets)
    # Filter the pickle file names further based on the selected market
    filtered_files = [filename for filename in filtered_files if get_country_and_market(filename.split('.')[0])[1] == market_option]
else:
    st.write('No models found for the selected country')

if filtered_files:
    # Create a dictionary to hold the models
    models = {}
    # Load the models from the pickle files
    for filename in filtered_files:
        with open(os.path.join(model_dir, filename), 'rb') as f:
            model = pickle.load(f)
            models[filename] = model
else:
    st.write('No models found for the selected market')

if filtered_files:
    # Get the model for the selected country and market
    model_name = filtered_files[0] # Assumes only one file is selected
    model = models[model_name]

    y = data.loc[(data['Country'] == country_option) & (data['Market'] == market_option), 'Count Of SR'].resample('MS').sum()
    
    # Ask for the number of months to forecast
    months = st.selectbox("Select the number of months to forecast:", [3, 6, 9, 12])
    
    if st.button('Make Forecast'):
        # print('*'*50)
        # print(f'Prediction for {months} months')
        pred_uc = model.get_forecast(steps=months)
        pred_ci = pred_uc.conf_int()
        pred_df = pd.DataFrame({'predicted_mean': pred_uc.predicted_mean,
                            'lower_bound': pred_ci.iloc[:, 0],
                            'upper_bound': pred_ci.iloc[:, 1]})
        st.write(f'<style>div.stTable td{{text-align: center!important}}</style>', unsafe_allow_html=True)
        st.table(pred_df)
        # st.table(pred_df.style.set_table_styles([{'selector': 'td', 
        # 'props': [('text-align', 'center')]}]))
        ax = y.plot(label='observed', figsize=(14, 7))
        pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
        ax.fill_between(pred_ci.index,
                        pred_ci.iloc[:, 0],
                        pred_ci.iloc[:, 1], color='k', alpha=.5)
        for date in pd.date_range(start=y.index[-1], periods=months, freq='MS'):
            ax.axvline(date, linestyle='--', color='k', alpha=0.2, )
        ax.axvline(linestyle='--', color='k',label=f'{months}-Month Forecast')
        ax.set_xlabel('Date')
        ax.set_ylabel('Count Of SR')
        plt.legend()
        
        graph_title = f'<h5><center> Forecast Graph of {months} months for Country : {country_option} & market : {market_option} </center> </h5>'
        st.markdown(graph_title, unsafe_allow_html=True)

        #st.write(f'Forecast Graph for Country {country_option} with market {market_option} for {months} months')
        st.pyplot() # Display the plot in the Streamlit app
