# schreib mir eine FastAPI App die mir die einen Wert zurückgibt und eine csv datei als Input nimmt
# und die Summer aller Werte in der Spalte "Value" zurückgibt

from fastapi import FastAPI
import pandas as pd

app = FastAPI()

@app.get("/sum")
def sum():
    df = pd.read_csv("Paul\input.csv")
    return df["Value"].sum()

# Path: Paul\input.csv
# Value
# 1
# 2
# 3
# 4
# 5

# Path: Paul\output.txt
# 15

# Path: Paul\output.json
# {
#   "Value": 15
# }

# Path: Paul\output.yaml
# Value: 15

# Path: Paul\output.xml
# <Value>15</Value>

# Path: Paul\output.html
# <html>
#   <body>
#     <h1>Value</h1>
#     <p>15</p>
#   </body>
# </html>

# Path: Paul\output.csv
# Value
# 15











