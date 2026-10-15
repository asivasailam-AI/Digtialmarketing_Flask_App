import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from nselib import capital_market as cm
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.statespace.varmax import VARMAX
import warnings

warnings.filterwarnings('ignore')

def clean_nselib_price_volume(df):
    """Standard cleaning function for nselib data"""
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%Y', errors='coerce')
    numeric_cols = ['PrevClose', 'OpenPrice', 'HighPrice', 'LowPrice', 'LastPrice', 
                    'ClosePrice', 'AveragePrice', 'TotalTradedQuantity', 
                    'Turnover₹', 'No.ofTrades']
    for col in numeric_cols:
        df[col] = df[col].astype(str).str.replace(',', '', regex=False).str.strip()
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.sort_values('Date').set_index('Date')
    return df

def run_deployment():
    print("\n====================================================")
    print("   📈 SHARE PRICE PREDICTION DEPLOYMENT APP 📈")
    print("====================================================\n")

    try:
        # 1. GET USER INPUTS
        symbol = input("Enter Company Symbol (e.g., ITCHOTELS): ").upper().strip()
        start_date = input("Enter Start Date (DD-MM-YYYY): ").strip()
        end_date = input("Enter End Date (DD-MM-YYYY): ").strip()
        days_to_predict = int(input("Enter number of days to predict: "))

        # 2. FETCH LATEST DATA
        print(f"\nFetching data for {symbol}...")
        raw_data = cm.price_volume_data(symbol=symbol, from_date=start_date, to_date=end_date)
        
        if raw_data is None or raw_data.empty:
            print("❌ No data found. Please check the symbol and dates.")
            return

        df_cleaned = clean_nselib_price_volume(raw_data)
        columns = ["OpenPrice", "HighPrice", "LowPrice", "ClosePrice"]
        data_selected = df_cleaned[columns].apply(pd.to_numeric, errors='coerce').dropna().reset_index(drop=True)

        if len(data_selected) < 5:
            print("❌ Insufficient data points to generate a forecast.")
            return

        # 3. LOAD PICKLED ASSETS
        print("Loading trained model assets...")
        with open("models/TSA_Share_Prediction_Model.pkl", "rb") as f:
            assets = pickle.load(f)
        
        model_fit = assets['model']
        scaler = assets['scaler']
        
        # 4. PREDICTION PIPELINE
        # Scale new company data
        scaled_data = scaler.transform(data_selected)
        scaled_df = pd.DataFrame(scaled_data, columns=columns)

        # --- CRITICAL FIX ---
        # Instead of calling forecast() on the pickled model directly, 
        # we must update the model's state with the new company's data.
        # We do this by using the 'append' method which adds new observations to the existing model.
        print(f"Updating model state with latest {symbol} data...")
        updated_model = model_fit.append(scaled_df.values)
        
        # Now forecast from the updated state
        forecast_scaled = updated_model.forecast(steps=days_to_predict)
        # ---------------------
        
        # Inverse Scale to get actual prices
        forecast_original = scaler.inverse_transform(forecast_scaled)
        
        # Create Results DataFrame
        forecast_df = pd.DataFrame(forecast_original, columns=columns)
        forecast_df.index = [f"Day {i+1}" for i in range(days_to_predict)]

        # 5. OUTPUT RESULTS
        print(f"\n✅ SUCCESS: Predicted Prices for {symbol} for the next {days_to_predict} days:")
        print("-" * 60)
        print(forecast_df.round(2))
        print("-" * 60)

        # 6. OPTIONAL VISUALIZATION
        plt.figure(figsize=(10, 5))
        plt.plot(forecast_df.index, forecast_df["ClosePrice"], marker='o', color='red', label="Predicted Close Price")
        plt.title(f"Forecasted Close Price for {symbol}")
        plt.xlabel("Days")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True)
        plt.show()

    except ValueError:
        print("❌ Invalid input. Please ensure dates are DD-MM-YYYY and days are a number.")
    except FileNotFoundError:
        print("❌ Model file 'TSA_Share_Prediction_Model.pkl' not found. Please run the notebook first.")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")

if __name__ == "__main__":
    run_deployment()