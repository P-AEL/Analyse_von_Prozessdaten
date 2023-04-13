"""test for the api"""
import requests


url = "http://127.0.0.1:8000/receive_data"
file_path = r"C:\Users\t-ehm\iCloudDrive\Studium\Data_Science_Semester4\Analyse_von_Prozess_und_Produktdaten\Analyse_von_Prozessdaten\Paul\Batch_181.csv"
file = {'file': open(file_path, 'rb')}
r = requests.post(url, files=file)
print(r.text)

# import pandas as pd

# df = pd.read_pickle(r"C:\Users\t-ehm\iCloudDrive\Studium\Data_Science_Semester4\Analyse_von_Prozess_und_Produktdaten\Analyse_von_Prozessdaten\setup\current_tank_data.pkl")

# print(df.columns)


