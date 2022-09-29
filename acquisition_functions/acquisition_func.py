import acquisition_functions
import torch


def acquisition_func(args, model, data, candidate_points, predictions, variance):
    """
    Get acquisition function based on the arguments
    :param args: arguments
    :param model:
    :param data:
    :param candidate_points: candidate points, for which the acquisition function should be evaluated on
    :param predictions: predictions for the candidate space
    :param variance:
    :return: acquisition function
    """

    crit = args.selection_criteria
    if crit in ['variance', "sequential", "sequential_relevant_variance", "parallel_relevant_variance", 'max_variance',
                'mean_variance', 'variance_lhs']:
        acq_func = acquisition_functions.variance.Variance(model, candidate_points, variance)

    elif crit == "random":
        acq_func = acquisition_functions.random.Random(model, candidate_points, args)

    elif crit == "mcmc_qbc":
        if args.model_type not in ['fbgp_mcmc']:
            raise NotImplementedError(f"Trying to use {crit} with the model: {args.model_type}."
                                      f" If you want apply {crit} use: fbgp_mcmc.")
        acq_func = acquisition_functions.bayesian_active_learning.BQBC(model, candidate_points)
    elif crit == "mcmc_mean_variance":
        if args.model_type not in ['fbgp_mcmc']:
            raise NotImplementedError(f"Trying to use {crit} with the model: {args.model_type}."
                                      f" If you want apply {crit} use: fbgp_mcmc.")
        acq_func = acquisition_functions.bayesian_active_learning.BALM(model, candidate_points)
    elif crit == "mcmc_gmm":
        if args.model_type not in ['fbgp_mcmc']:
            raise NotImplementedError(f"Trying to use {crit} with the model: {args.model_type}."
                                      f" If you want apply {crit} use: fbgp_mcmc.")
        acq_func = acquisition_functions.bayesian_active_learning.QBMGP(model, candidate_points)
    elif crit == "mcmc_bald":
        if args.model_type not in ['fbgp_mcmc']:
            raise NotImplementedError(f"Trying to use {crit} with the model: {args.model_type}."
                                      f" If you want apply {crit} use: fbgp_mcmc.")
        acq_func = acquisition_functions.bayesian_active_learning.BALD(model, candidate_points)
        # acq_func = acquisition_functions.bayesian_active_learning.BALD2(model, candidate_points)
    else:
        raise NotImplementedError(f"The acquisition function '{crit}' is not implemented. Change the "
                                  f"acquisition function with args.selection_criteria.")

    return acq_func
