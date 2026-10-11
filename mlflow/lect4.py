import mlflow
import pandas as pd
mlflow.set_experiment("youtube tutorial")

with mlflow.start_run(run_name="Logging Demo",run_id="edc00081cb25449eb5b10cd00dc0a6d0"):
  #paramters
  #key value
  #dictionary
  mlflow.log_params({
    "learning_rate":0.03
  })
  mlflow.log_params({
    "epoch":100

  })
  parameters={
    "learning_rate1":0.04,
    "epoch1":200
  }
  mlflow.log_params(parameters)
  mlflow.log_metric("accuracy", 0.90)
  metrics={
    "accuracy1": 80

  }
  mlflow.log_metrics(metrics)

  #artifacts
  artifact_path="images.png"
  mlflow.log_artifact(artifact_path)

  demo_df=pd.DataFrame({"name":["Tejaswini","Sindhu"]})
  titanic_df=pd.read_csv("titanic.csv")
  mlflow.log_table(titanic_df,"titanic.json")
  