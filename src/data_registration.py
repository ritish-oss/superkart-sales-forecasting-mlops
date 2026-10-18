import os, sys, pandas as pd
EXPECTED = ["Product_Id", "Product_Weight", "Product_Sugar_Content", "Product_Allocated_Area", "Product_Type", "Product_MRP", "Store_Id", "Store_Establishment_Year", "Store_Size", "Store_Location_City_Type", "Store_Type", "Product_Store_Sales_Total"]
if not os.path.exists("data/SuperKart.csv"):
    sys.exit(1)
df = pd.read_csv("data/SuperKart.csv")
missing = [c for c in EXPECTED if c not in df.columns]
if missing:
    sys.exit(1)
print(f"[DATA REGISTRATION SUCCESS] Shape: {df.shape}")
