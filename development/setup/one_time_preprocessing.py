import pandas as pd

df = pd.read_pickle(r"C:\Users\t-ehm\iCloudDrive\Studium\Data_Science_Semester4\Analyse_von_Prozess_und_Produktdaten\Analyse_von_Prozessdaten\setup\current_tank_data.pkl")

# timestamp to datetime
df.timestamp = pd.to_datetime(df.timestamp)

# ignore step 2
df_new = df.loc[df["CuStepNo ValueY"] != 2] #hat er uns ja gesagt diesen Schritt zu ignorieren

# create column with next step
df_new["Next_Step"] = df_new["CuStepNo ValueY"].shift()

# get all steps where the next step is different
df_steps = df_new.loc[df_new["CuStepNo ValueY"] != df_new["Next_Step"]]

#df_steps ist nen Hilfsdataframe um später die nötigen nummer von Batches auf die Originaldaten zu bekommen

# get all timestamps where the step is 1 (start of batch)
times = df_steps[df_steps["CuStepNo ValueY"] == 1].timestamp.to_list()
df_train = pd.DataFrame(columns=df_new.columns)

for i in range(times.__len__()-1):
    df_test1 = df_new.loc[(df_new.timestamp >=times[i])&(df_new.timestamp <times[i+1])]
    df_test1["Batch"] = i
    df_train = pd.concat([df_train,df_test1])