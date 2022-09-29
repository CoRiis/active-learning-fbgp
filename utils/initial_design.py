import numpy as np
import skopt


def sample_initial_inputs(n_samples, search_space, method='random', seed=None):
    """
    Sample initial data points

    :param n_samples: the number of initial data points to sample
    :param search_space: search space to sample data points from
    :param method: method used for sampling
    :return: initial data points in X values and y values
    """

    # array for the points x' and respective f(x') that we sample using the acquisition function
    x_sample = []

    if method == 'random':
        # if n_samples are smaller than the number of unique points, we get some distinct random samples
        # otherwise, just get some random samples
        n_unique = len(np.unique(search_space, axis=0))
        if n_samples < n_unique:
            unique_ss = np.unique(search_space, axis=0)
        else:
            unique_ss = search_space

        replacement = True if n_samples > unique_ss.shape[0] else False
        rng = np.random.default_rng(seed=seed)
        i = rng.choice(np.arange(unique_ss.shape[0]), size=n_samples, replace=replacement)
        x_sample = unique_ss[i]

    elif method == 'lhs':
        # Use LHS from scikit-optimize
        # https://scikit-optimize.github.io/stable/modules/generated/skopt.sampler.Lhs.html#skopt.sampler.Lhs
        # Create 'space' from a list of tuples with (min, max) values for each feature

        if len(search_space.shape) == 1:
            space = [(np.min(search_space, axis=0), np.max(search_space, axis=0))]
        else:
            space = [tuple(x) for x in list(np.array([np.min(search_space, axis=0), np.max(search_space, axis=0)]).T)]
        space = skopt.space.Space(space)
        lhs = skopt.sampler.Lhs(lhs_type="classic", criterion='maximin')
        x_sample = lhs.generate(space.dimensions, n_samples)
        if len(search_space.shape) == 1:
            x_sample = np.array(x_sample).squeeze(-1)

    return x_sample
