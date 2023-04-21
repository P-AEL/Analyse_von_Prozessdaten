from fastapi import FastAPI, File, UploadFile
import numpy as np
import pandas as pd

from sktime.classification.kernel_based import RocketClassifier
from tslearn.preprocessing import TimeSeriesResampler
from typing import Tuple


def get_data_to_analyse(incoming_data: pd.DataFrame) -> pd.DataFrame:
    """
    Returns the data that needs to be analysed.
    If the incoming data is a new batch, its the only batch that needs to be predicted on.
    If not, the saved data from the newest batch needs to be concatenated with the incoming data.
    :param incoming_data: data that is coming in
    :return: data that needs to be analysed
    """
    # importing the saved data from the newest batch
    newest_batch_data = pd.read_csv(r"C:\Users\t-ehm\iCloudDrive\Studium\Data_Science_Semester4\Analyse_von_Prozess_und_Produktdaten\Analyse_von_Prozessdaten\tim\newest_batch_data_v2.csv")

    if incoming_data.iloc[0]["CuStepNo ValueY"] == 1 and newest_batch_data.iloc[-1]["CuStepNo ValueY"] == 3:
        print("incoming data only contains new batch")
        incoming_data['new_batch'] = np.where((incoming_data["CuStepNo ValueY"] == 1) & (incoming_data["CuStepNo ValueY"].shift(1) == 3), 1, 0)
        incoming_data['Batch'] = incoming_data['new_batch'].cumsum() + newest_batch_data['Batch'].max() + 1
        data_to_analyse = incoming_data.drop(columns=["new_batch", 'Unnamed: 0'])
    else:
        print("concatenated data with newest_batch_data")
        incoming_data["new_batch"] = np.where((incoming_data["CuStepNo ValueY"] == 1) & (incoming_data["CuStepNo ValueY"].shift(1) == 3), 1, 0)
        incoming_data["Batch"] = incoming_data["new_batch"].cumsum() + newest_batch_data["Batch"].max()
        data_to_analyse = pd.concat([newest_batch_data, incoming_data.drop(columns=["new_batch", 'Unnamed: 0'])])
    
    print("data_to_analyse shape: ", data_to_analyse.shape)
    return data_to_analyse

def get_training_and_prediction_data(data_to_analyse: pd.DataFrame, tank_training_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates the training data according to the data to predict on.
    Training data is getting reduced to the same steps that are included in the data to predict on.
    For the newest step the amount of data points is reduced to the amount of data points in the data to predict on.
    :param data_to_analyse: data to predict on
    :param tank_training_data: training data
    :return: X_train, y_train and data to predict on
    """
    
    # get the last step of the data to predict on
    last_step = data_to_analyse["CuStepNo ValueY"].unique()[-1]

    # get the amount of data points in the last step of the data to predict on
    amount_of_data_points = data_to_analyse[data_to_analyse["CuStepNo ValueY"] == last_step].__len__()

    # delete all data points in the training data that are not in the last step of the data to predict on
    for i in tank_training_data["Batch"].unique():
        tank_training_data.drop(tank_training_data.loc[(tank_training_data["Batch"] == i)&(tank_training_data["CuStepNo ValueY"] == last_step)][amount_of_data_points:].index,inplace=True)

    # delete all the steps that are not in the data to predict on
    for i in tank_training_data["CuStepNo ValueY"].unique():
        if i not in data_to_analyse["CuStepNo ValueY"].unique():
            tank_training_data.drop(tank_training_data.loc[tank_training_data["CuStepNo ValueY"] == i].index,inplace=True)
    
    
    # get training data. X_train_data is a list of numpy arrays. Each numpy array contains the data points of one batch
    X_train_data = [tank_training_data.loc[tank_training_data["Batch"]==i].drop(columns=["DeviationID ValueY", "Batch", "CuStepNo ValueY", "Next_Step", "timestamp", "Unnamed: 0"]).to_numpy() for i in tank_training_data["Batch"].unique()]
    y_train_data = np.array([tank_training_data.loc[tank_training_data["Batch"]==i]["DeviationID ValueY"].unique()[0] for i in tank_training_data["Batch"].unique()])

    # Interpolation of training data to the same length as the data to predict on. For data_to_analyse one dimension is added
    print("Data points in Batch 200", tank_training_data.loc[tank_training_data["Batch"]==200].shape[0])
    print("Data points for mainleveltank in Batch 200 by step", tank_training_data.loc[(tank_training_data["Batch"]==200)].groupby("CuStepNo ValueY").count()["LevelMainTank ValueY"])
    print("Data points in data_to_analyse", data_to_analyse.shape[0])
    print("Data points for mainleveltank in data_to_analyse by step", data_to_analyse.groupby("CuStepNo ValueY").count()["LevelMainTank ValueY"])
    data_points_per_batch = int(np.median([i.shape[0] for i in X_train_data]))
    X_train_data = TimeSeriesResampler(sz=data_points_per_batch).fit_transform(X_train_data)
    print("data_points_per_batch: ",data_points_per_batch)
    print("X_train_data shape: ",X_train_data.shape)
    print("data_to_analyse shape: ",data_to_analyse.shape)
    # drop unnecessary columns of data_to_analyse
    data_to_analyse.drop(columns=["DeviationID ValueY", "CuStepNo ValueY", "timestamp", "Batch"], inplace=True)
    
    data_to_analyse = TimeSeriesResampler(sz=data_points_per_batch).fit_transform(np.expand_dims(data_to_analyse.to_numpy(), axis=0))
    
    # reshape X_train_data from (n_batches, n_data_points, n_features) to (n_batches, n_features, n_data_points)
    print("X_train shape: ",X_train_data.shape)
    X_train_data = np.transpose(X_train_data, (0, 2, 1))
    print("X_train shape after reshape",X_train_data.shape)
    print("data_to_analyse shape: ",data_to_analyse.shape)

    # reshape data_to_analyse from (n_data_points, n_features) to (1, n_features, n_data_points)
    data_to_analyse = np.transpose(data_to_analyse, (0, 2, 1))
    print("data_to_analyse shape after reshape: ",data_to_analyse.shape)

    return X_train_data, y_train_data, data_to_analyse

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "Worldfs"}

# api to receive csv data from user
@app.post("/receive_data")
async def receive_data(file: UploadFile = File(...)):
    """Receive tank data from user and return it as a pandas dataframe"""
    try:
        try:
            tank_training_data = pd.read_pickle(r"C:\Users\t-ehm\iCloudDrive\Studium\Data_Science_Semester4\Analyse_von_Prozess_und_Produktdaten\Analyse_von_Prozessdaten\development\setup\current_tank_data.pkl")
        except:
            tank_training_data = pd.read_pickle(r"/Users/timehmann/Library/Mobile Documents/com~apple~CloudDocs/Studium/Data_Science_Semester4/Analyse_von_Prozess_und_Produktdaten/Analyse_von_Prozessdaten/development/setup/current_tank_data.pkl")

        # Drop last row of batch 238 because of data error
        tank_training_data.drop(tank_training_data.loc[(tank_training_data.Batch == 238)&(tank_training_data["CuStepNo ValueY"] == 3)].tail(1).index,inplace=True)

        new_data = pd.read_csv(file.file)
        print("data_received")
        data_to_analyse = get_data_to_analyse(new_data)

        # Put each batch to predict on in a dictionary
        batches_to_predict_dict = {i: data_to_analyse.loc[data_to_analyse["Batch"] == i] for i in data_to_analyse["Batch"].unique()}

        # update the file with the newest batch data
        newest_batch_data = batches_to_predict_dict[max(batches_to_predict_dict.keys())]
        newest_batch_data.to_csv(r"C:\Users\t-ehm\iCloudDrive\Studium\Data_Science_Semester4\Analyse_von_Prozess_und_Produktdaten\Analyse_von_Prozessdaten\tim\newest_batch_data_v2.csv", index=False)

        # Train and predict the DeviationID for each batch
        for batch in batches_to_predict_dict.keys():
            X_train, y_train, data_to_analyse = get_training_and_prediction_data(batches_to_predict_dict[batch], tank_training_data)
            clf = RocketClassifier(num_kernels=300, n_jobs=-1)
            clf.fit(X_train, y_train)
            predicted_deviation_id = clf.predict(data_to_analyse)
            print(batch, predicted_deviation_id)
            # write the prediction of the batch in a csv file as a new row
            with open(r"C:\Users\t-ehm\iCloudDrive\Studium\Data_Science_Semester4\Analyse_von_Prozess_und_Produktdaten\Analyse_von_Prozessdaten\tim\batch_predictions.csv", "a") as f:
                f.write(f"\n{batch},{predicted_deviation_id[0]}")
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()

    return {"message": f"Successfully uploaded {file.filename}"}