"""test for the api"""
import requests
import csv
import time
import os


url = "http://127.0.0.1:8000/receive_data"
#file_path = r"C:\Users\t-ehm\iCloudDrive\Studium\Data_Science_Semester4\Analyse_von_Prozess_und_Produktdaten\Analyse_von_Prozessdaten\Paul\Batch_181.csv"
file_path = r"C:\Users\t-ehm\iCloudDrive\Studium\Data_Science_Semester4\Analyse_von_Prozess_und_Produktdaten\Analyse_von_Prozessdaten\tim\test_batch_76_181.csv"

# Open the CSV file and read the contents
with open(file_path, 'r') as f:
    reader = csv.reader(f)
    header = next(reader)  # Skip the header row
    rows = [row for row in reader]

# Calculate the number of rows in the CSV file
num_rows = len(rows)

# Calculate the number of rows to send in each request (10% of total)
batch_size = int(num_rows * 0.1)

# Send the CSV data to the API in batches
for i in range(0, num_rows, batch_size):
    # Select the next batch of rows to send
    batch = rows[i:i+batch_size]

    # Create a new CSV file with the batch of rows
    with open('batch_temp.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(batch)

    # Send the file data to the API using requests
    with open('batch_temp.csv', 'rb') as f:
        file_data = f.read()
    response = requests.post(url=url, files={'file': file_data})

    # Wait for 30 seconds before sending the next request
    time.sleep(3)

# Delete the temporary batch file
os.remove('batch_temp.csv')

# import pandas as pd

# df = pd.read_pickle(r"C:\Users\t-ehm\iCloudDrive\Studium\Data_Science_Semester4\Analyse_von_Prozess_und_Produktdaten\Analyse_von_Prozessdaten\setup\current_tank_data.pkl")

# print(df.columns)


