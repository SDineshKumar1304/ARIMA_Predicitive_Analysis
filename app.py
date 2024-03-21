import streamlit as st
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import pickle
import os

def load_data(file_path):
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    return df

def train_model(train_data):
    model = ARIMA(train_data['sales'], order=(1, 1, 1))  
    fitted_model = model.fit()
    return fitted_model

def save_model(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

def load_model(filename):
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    return model

def main():
    st.title('Sales Forecasting App')

    st.sidebar.title('Upload CSV File')
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=['csv'])

    if uploaded_file is not None:
        df = load_data(uploaded_file)

        # Split data into train and test
        train_data = df[:-7]  
        test_data = df[-7:]  

        # Train or load the model
        model_path = 'C:\\Users\\svani\\Downloads\\Manikandan\\model\\arima_model.pkl'
        if st.sidebar.button('Train Model'):
            st.subheader('Training Model...')
            model = train_model(train_data)
            st.success('Model training completed.')
            # Save the trained model
            save_model(model, model_path)
        else:
            if os.path.exists(model_path):
                # Load the pre-trained model
                model = load_model(model_path)
            else:
                st.error('Model not found. Please train the model first.')

        # Plot current data and forecasted data
        st.subheader('Sales Data and Forecast Plot')
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df.index, df['sales'], label='Actual Sales')
        if 'model' in locals():
            forecast = model.forecast(steps=7)
            ax.plot(test_data.index, forecast, label='Forecasted Sales', color='red')
        ax.set_xlabel('Date')
        ax.set_ylabel('Sales')
        ax.set_title('Sales Data and Forecast')
        ax.legend()
        st.pyplot(fig)

        # Display forecast
        if 'model' in locals():
            st.subheader('Sales Forecast')
            st.write("Forecasted Sales for Next 7 Days:")
            st.write(forecast)

if __name__ == '__main__':
    main()
