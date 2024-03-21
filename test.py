import streamlit as st
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import pickle
import os

def load_model(filename):
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    return model

def load_test_data(file_path):
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    return df

def main():
    st.title('Sales Forecasting App')

    # Load pre-trained model
    model_path = 'C:\\Users\\svani\\Downloads\\Manikandan\\model\\arima_model.pkl'
    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        st.error('Model not found. Please make sure to train the model first.')

    # Upload test CSV file
    uploaded_file = st.file_uploader("Upload Test CSV", type=['csv'])
    if uploaded_file is not None:
        test_data = load_test_data(uploaded_file)

        # Predict with the model
        if 'model' in locals():
            st.subheader('Making Forecasts...')
            forecast = model.forecast(steps=len(test_data))
            st.success('Forecasting completed.')

            # Display forecast
            st.subheader('Sales Forecast')
            st.write("Forecasted Sales:")
            st.write(pd.DataFrame(forecast, columns=['Sales'], index=test_data.index))

if __name__ == '__main__':
    main()
