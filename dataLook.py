import pandas as pd

def load_card_data(filename='magic_cards.csv'):
    return pd.read_csv(filename)

def inspect_data(df):
    print("DataFrame Info:")
    print(df.info())
    
    print("\nSample Data:")
    print(df.head(10).to_string())

    print("\nSample Data (Last Few Rows):")
    print(df.tail(10).to_string())
    
    print("\nUnique values in colorIdentity:")
    print(df['colorIdentity'].unique())
    
    print("\nUnique values in keywords:")
    print(df['keywords'].unique())

# Main workflow
df = load_card_data()
inspect_data(df)