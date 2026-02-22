import pyodbc
import pandas as pd
import os
print("INGEST working directory:", os.getcwd())
conn = pyodbc.connect(
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=MY-DIGITAL-WORL;"
    "DATABASE=AdventureWorks2025;"
    "Trusted_Connection=Yes;"
)

query = """
select Product.ProductID, Product.Name as ProductName, ProductCategory.Name as ProductCategoryName, 
ProductSubcategory.Name ProductSubcategoryName, 
Color, Class, Style, Product.ProductSubcategoryID,ProductCategory.Name as ProductCategoryName
from Production.Product
inner join Production.ProductSubcategory
	on Product.ProductSubcategoryID = ProductSubcategory.ProductSubcategoryID
inner join Production.ProductCategory
	on ProductCategory.ProductCategoryID = ProductSubcategory.ProductCategoryID
"""

df = pd.read_sql(query, conn)

df["text"] = df.apply(
    lambda r: f"Product {r.ProductName} of Category {r.ProductCategoryName} of Subcategory {r.ProductSubcategoryName} with Color {r.Color}",
    axis=1
)

df[["ProductID","text"]].to_csv("documents.csv", index=False)

df = pd.read_sql(query, conn)
print(df.head())