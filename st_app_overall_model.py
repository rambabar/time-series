# import itertools
import numpy as np
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import pickle
import streamlit as st

warnings.filterwarnings("ignore")
st.set_option('deprecation.showPyplotGlobalUse', False)

# Create the Streamlit app
st.markdown("<h1 style='text-align: center;'> Forecasting App (Overall model)</h1>", unsafe_allow_html=True)
#st.title("Forecasting App")

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

# Load the model
with open('model_overall.pkl', 'rb') as f:
    model = pickle.load(f)

months = st.selectbox("Select the number of months to forecast:", [3, 6, 9, 12]) 
if st.button('Make Forecast'):

    print(model.summary().tables[1])

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
    
    graph_title = f'<h5><center> Forecast Graph of {months} months </center> </h5>'
    st.markdown(graph_title, unsafe_allow_html=True)

    #st.write(f'Forecast Graph for Country {country_option} with market {market_option} for {months} months')
    st.pyplot() # Display the plot in the Streamlit app