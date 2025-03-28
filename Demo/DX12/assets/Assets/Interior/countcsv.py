filename = "interior.csv"

columns = [
    "materialID",
]

import pandas as pd

df = pd.read_csv(filename)

for column in columns:
    print(df[column].value_counts())
