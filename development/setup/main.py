# ?minutentakt. CSV Daten kommen von der Maschine => konkatiniert.
# Vielleicht schon mit teildaten anaylisieren.
# mit FASTAPI. preprozessing timestamp löschen
# vorhergesagte deviation_id. Vielleicht unvollständig.


import pandas as pd
import numpy as np
from tslearn.metrics import dtw

old_data = pd.read_pickle(r"current_tank_data.pkl")

def preprocessing_of_new_data(*,old_data = old_data, new_data):
    """preprocessing of new data and return it as a pandas dataframe.
    The preprocessing includes:
    - adding information about the current batch number
    - if the new data doesnt start with the start of a new step, add the rest of the data from the old data
    """
    # timestamp to datetime
    new_data.timestamp = pd.to_datetime(new_data.timestamp)

    # ignore step 2
    new_data = new_data.loc[new_data["CuStepNo ValueY"] != 2]

    # create column with next step (previous step actually)
    new_data["Next_Step"] = new_data["CuStepNo ValueY"].shift()

    # get all steps where the next step is different (previous step actually)
    df_steps = new_data.loc[new_data["CuStepNo ValueY"] != new_data["Next_Step"]]

    # get all timestamps where the step is 1 (start of batch)
    times = df_steps[df_steps["CuStepNo ValueY"] == 1].timestamp.to_list()

    # check if last step of old data is the same as first step of new data
    if old_data["CuStepNo ValueY"].iloc[-1] != new_data["CuStepNo ValueY"].iloc[0]:
        # if not, batch number of new data is incresed by 1 if the last step of old data is 4

        ### TODO add data of last step of old data to new data to complete step data

        
        if old_data["CuStepNo ValueY"].iloc[-1] == 4:
            batch_no_of_new_data = old_data["Batch"].iloc[-1] + 1
        # if not, batch number of new data is the same as the last batch number of the old data
        else:
            batch_no_of_new_data = old_data["Batch"].iloc[-1]
    # if yes, batch number of new data is the same as the last batch number of the old data
    else:
        batch_no_of_new_data = old_data["Batch"].iloc[-1]
    
    # add batch number to new data
    if len(times) == 0:
        new_data["Batch"] = batch_no_of_new_data
    elif len(times) == 1:
        new_data.loc[(new_data.timestamp <times[0]),"Batch"] = batch_no_of_new_data
        new_data.loc[(new_data.timestamp >=times[0]),"Batch"] = batch_no_of_new_data + 1
    for i in range(len(times)-1):
        new_data.loc[(new_data.timestamp <times[0]),"Batch"] = batch_no_of_new_data
        new_data.loc[(new_data.timestamp >=times[i])&(new_data.timestamp <times[i+1]),"Batch"] = i + batch_no_of_new_data


    return new_data

def predict_deviation_id(*,old_data = old_data, new_data):
    """predict the deviation id of the new data for each step per batch in the new data"""
    y_predicted = {}
    for k in new_data.Batch.unique():
        for l in new_data["CuStepNo ValueY"].unique():
            score = []
            for i in range(1,11):
                meanscore = []
                for j in old_data[(old_data["CuStepNo ValueY"] == 8)&(old_data["DeviationID ValueY"] == i)].Batch.unique():
                    # dtw zwischen df_test1_new gefiltert nach batch k und step l und old_data gefiltert nach batch j step l und deviation i
                    # pro batch k und step l wird die dtw mit allen batches j und deviation i im gleichen Step berechnet
                    meanscore.append(dtw(new_data[(new_data["CuStepNo ValueY"]==l)&(new_data["Batch"]==k)].drop(["DeviationID ValueY","Batch","Next_Step","timestamp"],axis=1),old_data[(old_data["DeviationID ValueY"]==i)&(old_data["CuStepNo ValueY"]==l)&(old_data["Batch"]==j)].drop(["DeviationID ValueY","Batch","Next_Step","timestamp"],axis=1)))
                # für die deviation i wird der minimale dtw-Wert gespeichert
                score.append(np.min(meanscore))
            # save prediciton for step l in batch k
            y_predicted[(k,l)] = np.argmin(score)+1
    return y_predicted
    



from fastapi import FastAPI, File, UploadFile

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "Worldfs"}

# api to receive csv data from user
@app.post("/receive_data")
async def receive_data(file: UploadFile = File(...)):
    """Receive tank data from user and return it as a pandas dataframe"""
    try:
        new_data = pd.read_csv(file.file)
        # preprocessing of new data
        new_data = preprocessing_of_new_data(new_data = new_data)

        # predict deviation id
        predicted_deviation_ids = predict_deviation_id(new_data = new_data)

        # print differences in columns of current data and new data
        print(set(old_data.columns) - set(new_data.columns))
        print(predicted_deviation_ids)
        # save new data as csv file
        # new_data.to_csv(r"C:\Users\t-ehm\iCloudDrive\Studium\Data_Science_Semester4\Analyse_von_Prozess_und_Produktdaten\Analyse_von_Prozessdaten\setup\new_tank_data.csv",index=False)
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()

    return {"message": f"Successfully uploaded {file.filename}"}