import mlflow 
mlflow.set_experiment("nested run demo")
with mlflow.start_run(run_name="Parent Run") as parent_run:
  print("Parent run:{parent_run.info.run_id}")
with mlflow.start_run(run_name="Child Run 1") as child_run1:
  print("Child run 1:{child_run1.info.run_id}")
with mlflow.start_run(run_name="Child Run 2") as child_run2:
  print("Child run 2:{child_run2.info.run_id}")
with mlflow.start_run(run_name="Child Run 3") as child_run3:
  print("Child run 3:{child_run3.info.run_id}")