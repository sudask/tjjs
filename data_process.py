from config import *

df = pd.read_csv("Credit.txt", sep=' ')
df.drop(columns="ID", inplace=True)

print(df.head())