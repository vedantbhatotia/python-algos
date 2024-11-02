import pandas as py
import numpy as num

# products = py.read_csv(r"C:\Users\Vedant\Desktop\Online Retail.xlsx");
products = py.read_csv(r"C:\Users\Vedant\Desktop\WA_Fn-UseC_-HR-Employee-Attrition.csv")

print(products[["Age","Department"]])
num_rows = products[products.isna().any(axis = 1)]
print(num_rows) 