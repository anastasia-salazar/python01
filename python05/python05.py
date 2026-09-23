import pandas as pd  
import numpy as np

## Assignment 1: DataFrame Creation and Indexing 
print("Assignment 1: DataFrame Creation and Indexing")

data = np.random.randint(1, 21, size=(6, 4))
df = pd.DataFrame(data, columns=['A', 'B', 'C', 'D'])
print(df)
df = df.set_index('A')
print(df)
print(df.index)

df = pd.DataFrame(np.random.randint(1, 21, size=(3, 3)),
                    columns=['A', 'B', 'C'],
                    index=['X', 'Y', 'Z'])
print(df)
print(df.loc['Y', 'B'])


## Assignment 2: DataFrame Operations 
print("\nAssignment 2: DataFrame Operations")

data = np.random.randint(1, 21, size=(5, 3))
df = pd.DataFrame(data, columns=['A', 'B', 'C'])
print(df)
df['NewColumn'] = df['A'] * df['B']
print(df)

data2 = np.random.randint(1, 21, size=(4, 3))
df2 = pd.DataFrame(data2, columns=['A', 'B', 'C'])
print(df2)
print(df2.sum(axis=0))
print(df2.sum(axis=1))

## Assignment 3: Data Cleaning 
print("\nAssignment 3: Data Cleaning")

data = np.random.randint(1, 21, size=(5, 3))
df = pd.DataFrame(data, columns=['A', 'B', 'C'])
print(df)
df.loc[1, 'A'] = np.nan
df.loc[3, 'B'] = np.nan
print(df)

df.fillna(df.mean(), inplace=True)
print(df)

data1 = np.random.randint(1, 21, size=(6, 4))
df1 = pd.DataFrame(data1, columns=['A', 'B', 'C', 'D'])
print(df1)
df1.loc[0, 'A'] = np.nan
df1.loc[2, 'C'] = np.nan
print(df1)

df1.dropna(inplace=True)
print(df1)


## Assignment 4: Data Aggregation 
print("\nAssignment 4: Data Aggregation")

categories = np.random.choice(['A', 'B', 'C'], size=10)
values= np.random.randint(1, 21, size=10)

df = pd.DataFrame({'Category': categories, 'Value': values})
print(df)

result = df.groupby('Category')['Value'].agg(['sum', 'mean'])
print(result)

products = np.random.choice(['Computer', 'Laptop', 'Tablet'], size=10)
categories = np.random.choice(['A', 'B', 'C'], size=10)
sales = np.random.randint(1, 100, size=10)

df = pd.DataFrame({'Product': products, 'Category': categories, 'Sales': sales})
print(df)

result = df.groupby('Category')['Sales'].sum()
print(result)


## Assignment 5: Merging DataFrames
print("\nAssignment 5: Merging DataFrames")

df1 = pd.DataFrame({'ID': [1, 2, 3], 'Name': ['Alice', 'Bob', 'Charlie']})
df2 = pd.DataFrame({'ID': [1, 2, 3], 'Age': [25, 30, 35]})
merged = pd.merge(df1, df2, on='ID')
print(df1)
print(df2)
print(merged)

df3 = pd.DataFrame({'Soup': ['Tomato', 'Chicken Noodle', 'Mushroom'], 'Calories': [80, 150, 120]})
df4 = pd.DataFrame({'Pumpkin': ['Jack-o-Lantern', 'Cookies', 'Pie'], 'Holidays': ['Halloween', 'Thanksgiving', 'Christmas']})
print(df3)
print(df4)
print(pd.concat([df3, df4], axis=0))
print(pd.concat([df3, df4], axis=1))


## Assignment 6: Time Series Analysis 
print("\nAssignment 6: Time Series Analysis")

dates = pd.date_range(start='2024-01-01', periods=90, freq='D')
df = pd.DataFrame({'Value': np.random.randint(1, 21, size=90)}, index=dates)

print(df) 
print(df.resample('ME').mean())

dates = pd.date_range(start='2021-01-01', end='2021-12-31', freq='D')
df = pd.DataFrame({'Value': np.random.randint(1, 21, size=len(dates))}, index=dates)
print(df)

print(df.rolling(window=7).mean())


## Assignment 7: MultiIndex DataFrame
print("\nAssignment 7: MultiIndex DataFrame")

index = pd.MultiIndex.from_tuples([
    ('A', 'X'), ('A', 'Y'),
    ('B', 'X'), ('B', 'Y') 
], names=['Category', 'Subcategory'])

df = pd.DataFrame({'Value': [10, 20, 30, 40]}, index=index)
print(df) 

print(df.loc['B'])
print(df.loc[('B', 'X')])



categories = np.random.choice(['A', 'B'], size=10)
subcategories = np.random.choice(['X', 'Y'], size=10)
values = np.random.randint(1, 21, size=10)

df = pd.DataFrame({'Category': categories, 'SubCategory': subcategories, 'Value': values})
print(df)
result = df.groupby(['Category', 'SubCategory'])['Value'].sum()
print(result)


## Assignment 8: Pivot Table 
print("\nAssignment 8: Pivot Table")

dates = np.random.choice(['2024-01-01', '2024-01-02', '2024-01-03'], size=15)
categories = np.random.choice(['A', 'B', 'C'], size=15)
values = np.random.randint(1, 21, size=15)
data = pd.DataFrame({'Date': dates, 'Category': categories, 'Value': values})
print(data)
pivot = pd.pivot_table(data, values='Value', index='Date', columns='Category', aggfunc='sum')
print(pivot)

year = np.random.choice([2024, 2025, 2026], size=18)
quarter = np.random.choice(['Q1', 'Q2', 'Q3', 'Q4'], size=18)
revenue = np.random.randint(100, 1001, size=18)
datas = pd.DataFrame({'Year': year, 'Quarter': quarter, 'Revenue': revenue})
print(datas)

pivots = pd.pivot_table(datas, values='Revenue', index='Year', columns='Quarter', aggfunc='mean')
print(pivots)


## Assignment 9: Applying Functions 
print("\nAssignment 9: Applying Functions")

data = np.random.randint(1, 21, size=(5, 3))
df = pd.DataFrame(data, columns=['Col1', 'Col2', 'Col3'])
print(df)
print(df.apply(lambda x: x*2))

data2 = np.random.randint(1, 21, size=(6, 3))
df2 = pd.DataFrame(data2, columns=['Col1', 'Col2', 'Col3'])
print(df2)
df2['Total'] = df2.apply(lambda row: row.sum(), axis=1)
print(df2)


## Assignment 10: Working with Text Data 

s = pd.Series(['Hello', 'World', 'Python', 'Data Science', 'Machine Learning'])
print(s)
print(s.str.upper())

d = pd.Series(['Bandana', 'Moose', 'Mojave', 'Coffee', 'Computer'])
print(d)
print(d.str.slice(0, 3))


