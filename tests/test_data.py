from src.data_loader import get_market_data

prices, returns = get_market_data()
print(f"Data shape: {prices.shape}")
print(f"Returns shape: {returns.shape}")
print(f"\nFirst few returns:\n{returns.head()}")
print(f"\nReturns summary:\n{returns.describe()}")
