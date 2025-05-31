import torch.optim as optim


def get_optimizer(model_parameters, config):
    optimizer_name = config["optimizer"]["name"]
    optimizer_params = config["optimizer"]["params"]
    optimizer_class = getattr(optim, optimizer_name)
    return optimizer_class(model_parameters, **optimizer_params)


def get_scheduler(optimizer, config):
    if config["scheduler"] == "StepLR":
        return optim.lr_scheduler.StepLR(optimizer, step_size=config["step_size"], gamma=config["gamma"])
    elif config["scheduler"] == "CosineAnnealingLR":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["num_epochs"])
    else:
        return None
