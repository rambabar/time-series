# time-series
Personal Project on Forecasting of Service request with country & market wise model or overall model. 
Here utilized Streamlit as frontend.

Steps:
Create Environment 

Run "pip install -r requirements.txt"

** Steps to create model file & run overall streamlit app

Run "python overall_model_creation.py"

Run "streamlit run st_app_overall_model.py"

** Steps to create all model files & run country & marketwise st app

Run "python models_creation_c_m.py"

Run "streamlit run streamlit_app_c_m.py"


If you run it daily it create zip file of previous day folder with date & time. 
It also delete zip if it is older than 7 days. 
