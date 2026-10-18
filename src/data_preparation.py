import os, pandas as pd
from sklearn.model_selection import train_test_split
os.makedirs("artifacts", exist_ok=True)
df = pd.read_csv("data/SuperKart.csv")
sugar_map = {'low sugar': 'Low Sugar', 'low_sugar': 'Low Sugar', 'Low Sugar': 'Low Sugar', 'regular': 'Regular', 'reg': 'Regular', 'Regular': 'Regular', 'no sugar': 'No Sugar', 'No Sugar': 'No Sugar'}
df['Product_Sugar_Content'] = df['Product_Sugar_Content'].map(sugar_map).fillna(df['Product_Sugar_Content'])
df['Product_Weight'] = df['Product_Weight'].fillna(df.groupby('Product_Type')['Product_Weight'].transform('median')).fillna(df['Product_Weight'].median())
mode_size = df.groupby('Store_Type')['Store_Size'].apply(lambda x: x.mode()[0] if not x.mode().empty else 'Medium').to_dict()
df['Store_Size'] = df['Store_Size'].fillna(df['Store_Type'].map(mode_size)).fillna('Medium')
df['Store_Age'] = 2026 - df['Store_Establishment_Year']
df_clean = df.drop(columns=[c for c in ["Product_Id", "Store_Id", "Store_Establishment_Year"] if c in df.columns])
train, test = train_test_split(df_clean, test_size=0.2, random_state=42)
train.to_csv("artifacts/train.csv", index=False)
test.to_csv("artifacts/test.csv", index=False)
print("[DATA PREPARATION SUCCESS] Artifacts saved.")
