## prediction using sktime rocket classifier

import numpy as np
import pandas as pd
import random

from sktime.classification.kernel_based import RocketClassifier
from tslearn.preprocessing import TimeSeriesResampler
from sklearn.metrics import accuracy_score



# load data
tank_training_data = pd.read_pickle(r"C:\Users\t-ehm\iCloudDrive\Studium\Data_Science_Semester4\Analyse_von_Prozess_und_Produktdaten\Analyse_von_Prozessdaten\development\setup\current_tank_data.pkl")

test_batch_numbers = random.sample(list(tank_training_data["Batch"].unique()), 78)

test_data = tank_training_data[tank_training_data["Batch"].isin(test_batch_numbers)]
train_data = tank_training_data[~tank_training_data["Batch"].isin(test_batch_numbers)]

test_data_step_changes = test_data.loc[test_data["CuStepNo ValueY"] != test_data["CuStepNo ValueY"].shift()]
train_data_step_changes = train_data.loc[train_data["CuStepNo ValueY"] != train_data["CuStepNo ValueY"].shift()]

# create X_train and y_train
# X_train_step_1 is a list of numpy arrays, each array is the data for step 1 for every batch
# X data is without DeviationID, Batch and CuStepNo, Next_step, timestamp and Unnamed: 0


### TRAINING DATA STEP 1
X_train_step_1 = [train_data.loc[(train_data['Batch'] == i) & (train_data["CuStepNo ValueY"] == 1)].drop(columns=["DeviationID ValueY", "Batch", "CuStepNo ValueY", "Next_Step", "timestamp", "Unnamed: 0"]).to_numpy() for i in train_data["Batch"].unique()]
X_test_step_1 = [test_data.loc[(test_data['Batch'] == i) & (test_data["CuStepNo ValueY"] == 1)].drop(columns=["DeviationID ValueY", "Batch", "CuStepNo ValueY", "Next_Step", "timestamp", "Unnamed: 0"]).to_numpy() for i in test_data["Batch"].unique()]
y_train_step_1 = train_data_step_changes[train_data_step_changes["CuStepNo ValueY"] == 1]["DeviationID ValueY"].to_numpy()
y_test_step_1 = test_data_step_changes[test_data_step_changes["CuStepNo ValueY"] == 1]["DeviationID ValueY"].to_numpy()

data_amount_per_batch_s1 = [len(i) for i in X_train_step_1]

X_train_s1 = TimeSeriesResampler(sz=int(np.median(data_amount_per_batch_s1))).fit_transform(X_train_step_1)
X_test_s1 = TimeSeriesResampler(sz=int(np.median(data_amount_per_batch_s1))).fit_transform(X_test_step_1)

# X_train_s1 = X_train_s1.reshape(X_train_s1.shape[0],39,X_train_s1.shape[1])
# X_test_s1 = X_test_s1.reshape(X_test_s1.shape[0],39,X_test_s1.shape[1])

clf = RocketClassifier(num_kernels=500) 
clf.fit(X_train_s1, y_train_step_1) 
y_pred = clf.predict(X_test_s1)

print(accuracy_score(y_test_step_1, y_pred))


### TRAINING DATA STEP 7
X_train_step_7 = [train_data.loc[(train_data['Batch'] == i) & (train_data["CuStepNo ValueY"] == 7)].drop(columns=["DeviationID ValueY", "Batch", "CuStepNo ValueY", "Next_Step", "timestamp", "Unnamed: 0"]).to_numpy() for i in train_data["Batch"].unique()]
X_test_step_7 = [test_data.loc[(test_data['Batch'] == i) & (test_data["CuStepNo ValueY"] == 7)].drop(columns=["DeviationID ValueY", "Batch", "CuStepNo ValueY", "Next_Step", "timestamp", "Unnamed: 0"]).to_numpy() for i in test_data["Batch"].unique()]
y_train_step_7 = train_data_step_changes[train_data_step_changes["CuStepNo ValueY"] == 7]["DeviationID ValueY"].to_numpy()
y_test_step_7 = test_data_step_changes[test_data_step_changes["CuStepNo ValueY"] == 7]["DeviationID ValueY"].to_numpy()

data_amount_per_batch_s7 = [len(i) for i in X_train_step_7]

X_train_s7 = TimeSeriesResampler(sz=int(np.median(data_amount_per_batch_s7))).fit_transform(X_train_step_7)
X_test_s7 = TimeSeriesResampler(sz=int(np.median(data_amount_per_batch_s7))).fit_transform(X_test_step_7)

# X_train_s7 = X_train_s7.reshape(X_train_s7.shape[0],39,X_train_s7.shape[1])
# X_test_s7 = X_test_s7.reshape(X_test_s7.shape[0],39,X_test_s7.shape[1])

clf = RocketClassifier(num_kernels=500)
clf.fit(X_train_s7, y_train_step_7)
y_pred = clf.predict(X_test_s7)

print(accuracy_score(y_test_step_7, y_pred))

### TRAINING DATA STEP 8
X_train_step_8 = [train_data.loc[(train_data['Batch'] == i) & (train_data["CuStepNo ValueY"] == 8)].drop(columns=["DeviationID ValueY", "Batch", "CuStepNo ValueY", "Next_Step", "timestamp", "Unnamed: 0"]).to_numpy() for i in train_data["Batch"].unique()]
X_test_step_8 = [test_data.loc[(test_data['Batch'] == i) & (test_data["CuStepNo ValueY"] == 8)].drop(columns=["DeviationID ValueY", "Batch", "CuStepNo ValueY", "Next_Step", "timestamp", "Unnamed: 0"]).to_numpy() for i in test_data["Batch"].unique()]
y_train_step_8 = train_data_step_changes[train_data_step_changes["CuStepNo ValueY"] == 8]["DeviationID ValueY"].to_numpy()
y_test_step_8 = test_data_step_changes[test_data_step_changes["CuStepNo ValueY"] == 8]["DeviationID ValueY"].to_numpy()

data_amount_per_batch_s8 = [len(i) for i in X_train_step_8]

X_train_s8 = TimeSeriesResampler(sz=int(np.median(data_amount_per_batch_s8))).fit_transform(X_train_step_8)
X_test_s8 = TimeSeriesResampler(sz=int(np.median(data_amount_per_batch_s8))).fit_transform(X_test_step_8)

# X_train_s8 = X_train_s8.reshape(X_train_s8.shape[0],39,X_train_s8.shape[1])
# X_test_s8 = X_test_s8.reshape(X_test_s8.shape[0],39,X_test_s8.shape[1])

clf = RocketClassifier(num_kernels=500)
clf.fit(X_train_s8, y_train_step_8)
y_pred = clf.predict(X_test_s8)

print(accuracy_score(y_test_step_8, y_pred))

### TRAINING DATA STEP 3
X_train_step_3 = [train_data.loc[(train_data['Batch'] == i) & (train_data["CuStepNo ValueY"] == 3)].drop(columns=["DeviationID ValueY", "Batch", "CuStepNo ValueY", "Next_Step", "timestamp", "Unnamed: 0"]).to_numpy() for i in train_data["Batch"].unique()]
X_test_step_3 = [test_data.loc[(test_data['Batch'] == i) & (test_data["CuStepNo ValueY"] == 3)].drop(columns=["DeviationID ValueY", "Batch", "CuStepNo ValueY", "Next_Step", "timestamp", "Unnamed: 0"]).to_numpy() for i in test_data["Batch"].unique()]
y_train_step_3 = train_data_step_changes[train_data_step_changes["CuStepNo ValueY"] == 3]["DeviationID ValueY"].to_numpy()
y_test_step_3 = test_data_step_changes[test_data_step_changes["CuStepNo ValueY"] == 3]["DeviationID ValueY"].to_numpy()

data_amount_per_batch_s3 = [len(i) for i in X_train_step_3]

X_train_s3 = TimeSeriesResampler(sz=int(np.median(data_amount_per_batch_s3))).fit_transform(X_train_step_3)
X_test_s3 = TimeSeriesResampler(sz=int(np.median(data_amount_per_batch_s3))).fit_transform(X_test_step_3)

# X_train_s3 = X_train_s3.reshape(X_train_s3.shape[0],39,X_train_s3.shape[1])
# X_test_s3 = X_test_s3.reshape(X_test_s3.shape[0],39,X_test_s3.shape[1])

clf = RocketClassifier(num_kernels=500)
clf.fit(X_train_s3, y_train_step_3)
y_pred = clf.predict(X_test_s3)

print(accuracy_score(y_test_step_3, y_pred))



# Daten kommen rein

tank_training_data = pd.read_pickle(r"C:\Users\t-ehm\iCloudDrive\Studium\Data_Science_Semester4\Analyse_von_Prozess_und_Produktdaten\Analyse_von_Prozessdaten\development\setup\current_tank_data.pkl")

# simulates new data coming in
new_data = pd.read_csv(r"C:\Users\t-ehm\iCloudDrive\Studium\Data_Science_Semester4\Analyse_von_Prozess_und_Produktdaten\Analyse_von_Prozessdaten\Paul\Batch_76.csv")

# Daten aus dem aktuellen Batch ggf. aus Legacy Batch holen und preprocessing durchführen
if new_data.iloc[0]["CuStepNo ValueY"] == 1 and tank_training_data.iloc[-1]["CuStepNo ValueY"] == 3:
    data_to_analyse = new_data
else:
    newest_batch = tank_training_data.iloc[-1]["Batch"]
    data_to_analyse = pd.concat([tank_training_data[tank_training_data["Batch"] == newest_batch], new_data])
    tank_training_data = tank_training_data[tank_training_data["Batch"] != newest_batch]

# Aus den Einkommenden Daten wird der letzte Step genommen und entsprechend der Anzahl an überlieferten Datenpunkten aus diesem Step
# wird in den Batches der Trainingsdaten für diesen Step die gleiche Anzahl an Datenpunkten genommen und der Rest gelöscht
for i in tank_training_data["Batch"].unique():
    tank_training_data.drop(tank_training_data.loc[(tank_training_data["Batch"] == i)&(tank_training_data["CuStepNo ValueY"] == data_to_analyse["CuStepNo ValueY"].unique()[-1])][data_to_analyse[data_to_analyse["CuStepNo ValueY"] == data_to_analyse["CuStepNo ValueY"].unique()[-1]].__len__():].index,inplace=True)

# Trainingsdaten erstellen
X_train = [tank_training_data.loc[tank_training_data["Batch"]==i].drop(columns=["DeviationID ValueY", "Batch", "CuStepNo ValueY", "Next_Step", "timestamp", "Unnamed: 0"]).to_numpy() for i in tank_training_data["Batch"].unique()]
y_train = np.array([tank_training_data.loc[tank_training_data["Batch"]==i]["DeviationID ValueY"].unique()[0] for i in tank_training_data["Batch"].unique()])

# Interpolation für gleiche Länge
data_amount_per_batch = [len(i) for i in X_train]
X_train = TimeSeriesResampler(sz=int(np.median(data_amount_per_batch))).fit_transform(X_train)

# X_train = X_train.reshape(X_train.shape[0],39,X_train.shape[1])

# Training
clf = RocketClassifier(num_kernels=500)
clf.fit(X_train, y_train)

# Vorhersage

# Daten für Vorhersage vorbereiten
# TODO WENN ALTE BATCHES HINZUGEFÜGT WURDEN; MÜSSEN NOCH ANDERE COLUMNS GELÖSCHT WERDEN
data_to_analyse.drop(columns=["DeviationID ValueY", "CuStepNo ValueY", "timestamp", "Unnamed: 0"], inplace=True)

print(clf.predict(data_to_analyse.to_numpy().reshape(1,39,data_to_analyse.__len__())))

#print difference between columns in train and predict
#print(set(tank_training_data.columns) - set(data_to_analyse.columns))
#print(set(data_to_analyse.columns) - set(tank_training_data.columns))