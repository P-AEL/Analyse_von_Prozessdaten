import pandas as pd
import os.path

file_name= "SmA-Four-Tank-Batch-Process_V2.csv"

if not os.path.isfile(file_name):
    print("Missing file {}.".format(file_name))
    exit()
df= pd.read_csv(file_name, sep= ";")

if df.columns.isin(["DeviationID ValueY", "timestamp"]).any():
    df["timestamp"]= pd.to_datetime(df["timestamp"], format= "%Y-%m-%d %H:%M:%S")
    print("Found Data, Number of deviations: {}".format(len(df["DeviationID ValueY"].unique())))