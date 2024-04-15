import mlflow
import numpy as np

#mlflow.set_experiment("search-run-guide")


all_runs = mlflow.search_runs(search_all_experiments=True)

runs = mlflow.search_runs(
    filter_string='tags.mlflow.runName = "1"'
)

run_id = runs['run_id'].values[0]
print(run_id, type(run_id))

mlflow.start_run(experiment_id=1,
                 run_name='1',
                 run_id=run_id
                 ) 

mlflow.log_metrics(
            {
                'thing' : 23
            }, step = 2)

quit()

accuracy = np.arange(0, 1, 0.1)
loss = np.arange(1, 0, -0.1)
log_scale_loss = np.log(loss)
f1_score = np.arange(0, 1, 0.1)

batch_size = [2] * 5 + [4] * 5
learning_rate = [0.001, 0.01] * 5
model = ["GPT-2", "GPT-3", "GPT-3.5", "GPT-4"] + [None] * 6

task = ["classification", "regression", "causal lm"] + [None] * 7
environment = ["notebook"] * 5 + [None] * 5

dataset_name = ["custom"] * 5 + ["also custom"] * 5
dataset_digest = ["s8ds293b", "jks834s2"] + [None] * 8
dataset_context = ["train"] * 5 + ["test"] * 5

for i in range(10):
    with mlflow.start_run():
        mlflow.log_metrics(
            {
                "loss": loss[i],
                "accuracy": accuracy[i],
                "log-scale-loss": log_scale_loss[i],
                "f1 score": f1_score[i],
            }
        )

        mlflow.log_params(
            {
                "batch_size": batch_size[i],
                "learning rate": learning_rate[i],
                "model": model[i],
            }
        )

        mlflow.set_tags(
            {
                "task": task[i],
                "environment": environment[i],
            }
        )

        dataset = mlflow.data.from_numpy(
            features=np.random.uniform(size=[20, 28, 28, 3]),
            targets=np.random.randint(0, 10, size=[20]),
            name=dataset_name[i],
            digest=dataset_digest[i],
        )
        mlflow.log_input(dataset, context=dataset_context[i])