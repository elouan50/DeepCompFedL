import wandb

api = wandb.Api()

runs = api.runs("elouan50-rwth-aachen-university/deepcompfedl-pruning-layer-or-model")

for run in runs:
    
    run_name = run.name
    
    if run_name[6] == "T":
        run.config["layer-compression"] = True
    elif run_name[6] == "F":
        run.config["layer-compression"] = False
    elif run_name[7] == "T":
        run.config["layer-compression"] = True
    elif run_name[7] == "F":
        run.config["layer-compression"] = False
    run.update()
    
    
    # wandb.init(project="deepcompfedl-resnet12-cifar-10-r100",
    #            id=run_id,
    #            config=run_config,
    #            reinit=True)
    # for i, data in enumerate(run_data["accuracy"], 1):
    #     wandb.log({"accuracy": data}, step=i)