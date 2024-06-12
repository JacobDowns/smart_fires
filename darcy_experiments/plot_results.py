import mlflow
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter

#client = MlflowClient()
from mlflow import MlflowClient
client = MlflowClient()

def get_experiment(name, ax):

    # Get experiment id
    experiment_id = mlflow.get_experiment_by_name(name).experiment_id


    # Get existing or create new run
    runs =  mlflow.search_runs(
        [experiment_id], 
    )



    for ix, run in runs.iterrows():
        run_id = run['run_id']

        run_data = mlflow.get_run(run_id)
        pde_weight = float(run_data.data.params['pde_weight'])
        training_examples = int(run_data.data.params['training_examples'])
      

        color_weight = pde_weight

        print(pde_weight, training_examples)

        data = client.get_metric_history(run_id, 'validation_loss')


        times = []
        steps = []
        #steps = np.arange(301)
        vals = np.zeros(300)

        linestyles = {
            750 : 'solid',
            500 : 'dashed',
            250 : 'dotted'
        }

        cmap = plt.get_cmap('seismic', 8)

        for i in range(len(data)):
            step = data[i].step - 1
            vals[step] = data[i].value
        
        ax.plot(vals[1:], linestyle = linestyles[training_examples], color=cmap(color_weight), label=f'{pde_weight}, {training_examples}', linewidth=2)

    ax.set_title(name)
    ax.legend()
    ax.set_yscale('log') 
    ax.set_xlim([0., 300.])
    ax.set_ylim([2e-3, 0.065])

    num_labels = 20  # Set the desired number of labels
    ticks = np.logspace(np.log10(ax.get_yticks()[0]), np.log10(ax.get_yticks()[-1]), num_labels)
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FixedFormatter([f"{x:.3f}" for x in ticks]))

    ax.set_xlabel('Epochs')
    ax.set_ylabel('MSE (Log Scale)')

    ax.grid(True, alpha=0.3)
    #run_data_dict = mlflow.get_run(run_id).data.to_dictionary()
    #print(run_data_dict)

fig, axes = plt.subplots(1,4, figsize=(18, 9))
 
get_experiment('strong', axes[0])
get_experiment('weak', axes[1])
get_experiment('variational', axes[2])
get_experiment('data', axes[3])

plt.tight_layout()
plt.savefig('results.png', dpi=500)
plt.show()