import pandas as pd
from sklearn.linear_model import LinearRegression

#CSV files need to be change to 2 different sets that only contains 3DBall env
df = pd.read_csv("../data/combined_results.csv") #Train set
df_test = pd.read_csv("../data/combined_results.csv") #Test set

df["steps_to_threshold"] = pd.to_numeric(df["steps_to_threshold"], errors="coerce")
df["time_to_threshold"] = pd.to_numeric(df["time_to_threshold"], errors="coerce")

df["run_reached_threshold"] = (
    (df["steps_to_threshold"] > 0) &
    (df["time_to_threshold"] > 0)
).astype(int)

#Keep only those reached the threshold
def numeric(df):
    df_3dball = df[df["environment"] == "3DBall"]
    df_final = df_3dball[
    (df_3dball["steps_to_threshold"].notna()) & (df_3dball["steps_to_threshold"] > 0) &
    (df_3dball["time_to_threshold"].notna())  & (df_3dball["time_to_threshold"] > 0)
    ].copy() #the train set
    return df_final

df_train = numeric(df)

def ini_X(df):
    feature_cols = [
    "steps",
    "learning_rate",
    "batch_size",
    "buffer_size",
    "early_reward_mean",
    "p_loss_mean",
    "v_loss_mean",
    "entropy_mean",
    ]
    return df[feature_cols]

def predict(df_train, df_test, num):
    #Initialize X and Y
    X = ini_X(df_train)
    y = df_train["steps_to_threshold"]
    if (num == 1):
        y = df_train["time_to_threshold"]
    X_test = ini_X(df_test)

    #Linear Regression
    model_steps = LinearRegression()
    model_steps.fit(X, y)
    y_pred = model_steps.predict(X_test)
    return y_pred

y_steps_pred = predict(df_train, df_test, 0)
df_test["pred_steps_to_threshold"] = y_steps_pred

y_time_pred = predict(df_train, df_test, 1)
df_test["pred_time_to_threshold"] = y_time_pred

df_test.to_csv("test_with_predictions.csv", index=False)
