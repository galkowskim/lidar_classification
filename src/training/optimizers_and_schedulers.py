import torch.optim as optim

SCHEDULERS = {
    "StepLR": optim.lr_scheduler.StepLR,
    "CosineAnnealingLR": optim.lr_scheduler.CosineAnnealingLR,
    "OneCycleLR": optim.lr_scheduler.OneCycleLR,
}


def get_optimizer(model_parameters, config):
    optimizer_name = config["optimizer"]["name"]
    optimizer_params = config["optimizer"]["params"]
    optimizer_class = getattr(optim, optimizer_name)
    return optimizer_class(model_parameters, **optimizer_params)


def get_scheduler(optimizer, config):
    scheduler_name = config["scheduler"]["name"]
    scheduler_params = config["scheduler"]["params"][scheduler_name]

    if scheduler_name in SCHEDULERS:
        scheduler_class = SCHEDULERS[scheduler_name]
        return scheduler_class(optimizer, **scheduler_params)
    else:
        raise ValueError(f"Scheduler '{scheduler_name}' is not registered in SCHEDULERS.")
