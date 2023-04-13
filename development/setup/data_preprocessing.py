import pandas as pd
import datetime 
from data_load import df

df_new = df.loc[df["CuStepNo ValueY"] != 2]
df_new["Next_Step"] = df_new["CuStepNo ValueY"].shift()
df_steps = df_new.loc[df_new["CuStepNo ValueY"] != df_new["Next_Step"]]

df_steps["time_since_start"] = 0
steps = df_steps.timestamp.unique()
for i in range(len(steps)-1):
    df_steps.loc[df_steps.timestamp == steps[i],"time_since_start"] = (df_new.loc[(df_new.timestamp >= steps[i])&(df_new.timestamp < steps[i+1])].timestamp -steps[i]).max()
df_steps.loc[df_steps.time_since_start == 0,"time_since_start"] = datetime.timedelta(0)

times = df_steps[df_steps["CuStepNo ValueY"] == 1].timestamp.to_list()
df_test2 = pd.DataFrame(columns=df_new.columns)

for i in range(times.__len__()-1):
    df_test1 = df_new.loc[(df_new.timestamp >=times[i])&(df_new.timestamp <times[i+1])].groupby("CuStepNo ValueY",as_index=False,sort=False).mean()
    df_test1["Batch"] = i
    df_test2 = pd.concat([df_test2,df_test1])

df_test1 = df_new.loc[(df_new.timestamp >=times[261])].groupby("CuStepNo ValueY",as_index=False,sort=False).mean()


df_test2["time_since_start"] = df_steps[:-3].time_since_start.values
df_test2["time_since_start"] = df_test2["time_since_start"].dt.total_seconds()