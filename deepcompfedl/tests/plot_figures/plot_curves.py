
import numpy as np
import pandas as pd
import wandb
import matplotlib.pyplot as plt

api = wandb.Api()
nrounds = 99

baseline = np.zeros((nrounds))
accuracy = np.zeros((nrounds))

t_select = np.zeros((nrounds))
t_train = np.zeros((nrounds))
t_compress = np.zeros((nrounds))
t_decompress = np.zeros((nrounds))
t_aggregate = np.zeros((nrounds))

runs1 = [
    "elouan50-rwth-aachen-university/deepcompfedl-scenario1/runs/68ct5a6n",
    "elouan50-rwth-aachen-university/deepcompfedl-scenario1/runs/mi2eyjwk",
    "elouan50-rwth-aachen-university/deepcompfedl-scenario1/runs/vf3dv24i"
    ]
runs4 = [
    "elouan50-rwth-aachen-university/deepcompfedl-scenario4/runs/ro86agen",
    "elouan50-rwth-aachen-university/deepcompfedl-scenario4/runs/usdys3ol",
    "elouan50-rwth-aachen-university/deepcompfedl-scenario4/runs/e00z992d"
    ]
baselines1 = [
    "elouan50-rwth-aachen-university/deepcompfedl-scenario1/runs/iyymzle3",
    "elouan50-rwth-aachen-university/deepcompfedl-scenario1/runs/wqqu4evw",
    "elouan50-rwth-aachen-university/deepcompfedl-scenario1/runs/g6us7smf"
    ]


for run_path in runs1:
    
    run = api.run(run_path)
    df = run.history(keys=['accuracy', 't_select', 't_train', 't_compress', 't_decode', 't_aggregate'])

    accuracy += np.array(df['accuracy'])/3
    
    t_select += np.array(df['t_select'])/3
    t_train += np.array(df['t_train'])/3
    t_compress += np.array(df['t_compress'])/3
    t_decompress += np.array(df['t_decode'])/3
    t_aggregate += np.array(df['t_aggregate'])/3

for run_path in baselines1:
    
    run = api.run(run_path)
    df = run.history(keys=['accuracy'])

    baseline += np.array(df['accuracy'])/3

steps = np.array([i for i in range(1, nrounds+1)])

print(f"Total time per round: {np.average(t_select[1:])+np.average(t_train)+np.average(t_compress[1:])+np.average(t_decompress[1:])+np.average(t_aggregate[:-1]):.5f}s")


# Plot the accuracy values
plt.figure(figsize=(10, 5))
plt.plot(steps, accuracy, label='γ=0.5 & q=32', color='r')
plt.plot(steps, baseline, label='baseline', color='b')

# Labels and title
plt.xlabel('Round')
plt.xlim(0, nrounds)
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.title('Training Accuracy Across Rounds')
plt.legend()
plt.grid()

plt.show()
