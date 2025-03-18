import pandas as pd

# List of stocks to filter
stocks_to_filter = [
    'ADANIENT', 'ASIANPAINT', 'AXISBANK', 'BAJAJ-AUTO', 'BAJAJFINSV',
    'BAJFINANCE', 'BPCL', 'BRITANNIA', 'CIPLA', 'COALINDIA',
    'DIVISLAB', 'EICHERMOT', 'GRASIM', 'HCLTECH', 'HDFCBANK',
    'HDFCLIFE', 'HEROMOTOCO', 'HINDALCO', 'HINDUNILVR', 'ICICIBANK',
    'INDUSINDBK', 'INFY', 'ITC', 'JSWSTEEL', 'KOTAKBANK',
    'LT', 'M&M', 'MARUTI', 'NTPC', 'ONGC',
    'POWERGRID', 'RELIANCE', 'SBILIFE', 'SBIN', 'SUNPHARMA',
    'TATACONSUM', 'TATAMOTORS', 'TCS', 'TECHM', 'ULTRACEMCO',
    'UPL', 'WIPRO'
]

# Read the CSV file
df = pd.read_csv('c:/Users/KIIT/Desktop/gnn/new_model/data/maindata.csv')

# Filter the dataframe to include only the specified stocks
filtered_df = df[df['ticker'].isin(stocks_to_filter)]

# Save the filtered data to a new CSV file
filtered_df.to_csv('c:/Users/KIIT/Desktop/gnn/new_model/data/filtered_stocks.csv', index=False)

print(f"Filtered {len(stocks_to_filter)} stocks and saved to filtered_stocks.csv")