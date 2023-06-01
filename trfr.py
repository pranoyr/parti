from accelerate import Accelerator

# Tell the Accelerator object to log with wandb
accelerator = Accelerator(log_with="wandb")

# Initialise your wandb run, passing wandb parameters and any config information
accelerator.init_trackers(
    project_name="my_project", 
    config={"dropout": 0.1, "learning_rate": 1e-2},
    init_kwargs={"wandb": {"entity": "my-wandb-team"}}
    )


# Log to wandb by calling `accelerator.log`, `step` is optional
accelerator.log({"train_loss": 1.12, "valid_loss": 0.8}, step=1)
