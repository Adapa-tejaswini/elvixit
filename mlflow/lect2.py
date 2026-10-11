import mlflow
mlflow.set_experiment("demo")
with mlflow.start_run():
  mlflow.log_params({
    "learning_rate": 0.01,
    "epochs": 10
  })

  mlflow.log_metric("accuracy", 0.91)
  