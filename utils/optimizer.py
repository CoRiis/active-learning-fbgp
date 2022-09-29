# Optimizer for Gaussian Processes (GP)
import torch
import gpytorch


def get_optimizer(args, model, num_data=None):
    """
    Returns the optimizer

    :param args: arguments
    :param model: model
    :param num_data: number of data points in the training set
    :returns: optimizer and learning rate scheduler
    """
    hyperparameter_optimizer = torch.optim.Adam(model.parameters(), lr=args.initial_lr)
    opt = (hyperparameter_optimizer, None)

    sched = None
    if args.milestones is not None:
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(hyperparameter_optimizer, factor=0.5, patience=100,
                                                           verbose=False)

    return opt, sched
