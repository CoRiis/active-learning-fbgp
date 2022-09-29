import numpy as np
import torch
import random

from utils.transformations import transform
from acquisition_functions.acquisition_func import acquisition_func


def get_query(args, new_points, data, k_samples=1, repeat_sampling=1, seed=0):
    """
    Get the labels to the query from the oracle

    :param args: arguments
    :param new_points: np.array of possible new points (starts querying from left)
    :param search_space: search space (i.e. x)
    :param oracle_labels: oracle labels corresponding to search space (i.e. y)
    :param pool_labeled: list of indices of inputs from the search space, which have already been used (queried)
    :param k_samples: no. of distinct input values
    :param beta_sampling: the distance between distinct input values [0,1) (relative distance, e.g 10% of the range)
    :param repeat_sampling: no. of simulations for each input value
    :param seed: seed
    """

    # Get label from oracle
    if seed:
        random.seed(seed)  # for the data sets
        np.random.seed(seed)  # for the simulator

    pool_labeled = new_points[0, :].reshape(-1, new_points.shape[1])
    new_y = data.oracle.query(pool_labeled)

    pool_labeled = [pool_labeled] if isinstance(pool_labeled, np.int64) else pool_labeled
    pool_labeled = [pool_labeled] if isinstance(pool_labeled, np.float64) else pool_labeled

    data.pool_labeled = pool_labeled
    data = add_queried_datapoints(args, new_y, data, k_samples, repeat_sampling)
    return data


def add_queried_datapoints(args, new_y, data, k_samples, repeat_sampling):
    new_x = np.repeat(data.pool_labeled[:k_samples], repeat_sampling, axis=0)
    data.pool_labeled = None

    if data.train.x is None:
        data.train.x = torch.FloatTensor(new_x)
        data.train.y = torch.FloatTensor(new_y)
    else:
        data.train.x = torch.cat((data.train.x, torch.FloatTensor(new_x)))
        data.train.y = torch.cat((data.train.y, torch.FloatTensor(new_y)))
    return data


def do_repeat_sampling(args, data, repeat_sampling, new_points, search_space, oracle_labels, pool_labeled):
    """
    Get multiple samples with the same input value

    :param args: arguments
    :param repeat_sampling: the number of simulations to do at the input value
    :param new_points: np.array of possible new points (starts querying from left)
    :param search_space: search space (i.e. x)
    :param oracle_labels: oracle labels corresponding to search space (i.e. y)
    :param pool_labeled: list of indices of inputs from the search space, which have already been used (queried)
    """
    new_y = []
    for _ in range(repeat_sampling):
        # Using a simulator
        tmp_y = data.oracle.query(new_points[0].reshape(-1, new_points.shape[1]))
        new_y = np.concatenate((np.array(new_y), tmp_y))

    return new_y, pool_labeled


def index_descending(tensor):
    """
    Returns the order of points to query next
    """
    tensor = tensor if len(tensor.shape) == 1 else torch.mean(tensor, dim=1)
    return torch.argsort(tensor, descending=True)


def get_sorted_unlabeled_data_points(args, selection_array, search_space):
    """
    - Get indices of the selection array sorted in descending order
    - Extract new_points from the search_space using the indices

    :param args:
    :param selection_array:
    :param search_space:
    :return:
    """

    # Get the sequence of possible new data points
    new_points_index = index_descending(selection_array)
    new_points = search_space[new_points_index]

    if args.simulator == "motorcycle":
        new_points_index = np.array([np.where(search_space == tmp_x)[0][0] for tmp_x in new_points]).reshape(-1, 1)
        new_points = new_points_index

    return new_points, new_points_index


def compute_sample_strategy(args, model, data, predictions):
    """
    Applies a sampling strategy/criteria and returns a tensor with a value for each point in the search space.
    All strategies are defined such that next point to query has the highest value, e.g. argmax(selection_array).

    :param args: arguments
    :param model: gpytorch model
    :param data: class MyDataLoader
    :param predictions: dictionary with predictions, means and stddevs
    """

    output = {}
    selection_criteria = args.selection_criteria
    ss_unique = data.candidate_points

    variance = predictions['stddev']**2

    ss_unique_trans, _, _ = transform(torch.Tensor(ss_unique), data.x_mu, data.x_sigma, method=args.transformation_x)

    # Apply acquisition function
    torch.set_num_threads(args.num_chains)
    acq_func = acquisition_func(args, model, data, ss_unique_trans, predictions, variance)
    selection_array = acq_func.evaluate(x=None)
    torch.set_num_threads(1)

    new_points, new_points_index = get_sorted_unlabeled_data_points(args, selection_array, ss_unique)

    # Extract individual predictions for plotting
    if selection_criteria in ['mcmc_bald', 'mcmc_gmm', 'mcmc_mean_variance', 'mcmc_qbc']:
        output['batch_model_output'] = acq_func.predictions

    # Collect all outputs in a dict
    output['selection_array'] = selection_array
    output['new_points'] = new_points

    return output


def apply_active_learning_strategy(args, model, data, predict_output_querying, i):
    sample_strategy_output = compute_sample_strategy(args, model, data, predictions=predict_output_querying)
    selection_array = sample_strategy_output['selection_array']
    new_points = sample_strategy_output['new_points']

    # Get query and add it to data
    data = get_query(args, new_points, data,
                     k_samples=args.k_samples,
                     repeat_sampling=args.repeat_sampling,
                     seed=False)

    return sample_strategy_output, selection_array, new_points, data
