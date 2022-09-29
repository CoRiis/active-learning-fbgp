# Transformations
import torch


def transform(x, mu=None, sigma=None, method='identity', inverse=False, test_data=None):
    """
    
    :param x: tensor to be transformed
    :param mu: mean value
    :param sigma: standard deviation
    :param method: method to to transform
    :param inverse: should we do the inverse transformation? Then please prove mu and sigma
    """
    
    if method == 'standardize':
        return standardize(x, mu_x=mu, sigma_x=sigma, inverse=inverse)
    
    elif method == 'min_max_feature_scaling':
        return min_max_feature_scaling(x, min_x=mu, range_x=sigma, inverse=inverse)
    
    elif method == 'minusone_one_feature_scaling':
        return minusone_one_feature_scaling(x, min_x=mu, range_x=sigma, inverse=inverse)
    
    else: 
        return identity_transformation(x, inverse=inverse)
    
    
def identity_transformation(x, inverse=False):
    """
    Identity transformation
    
    :param x: tensor to be transformed
    :param inverse: inverse transformation
    """
    if inverse:
        return x
    
    mu_x = 0
    sigma_x = 1
    
    return x, mu_x, sigma_x
       

def standardize(x, mu_x=None, sigma_x=None, inverse=False):
    """
    Transform the data to have zero mean and a standard deviation at 1, e.g. z-score.
    
    :param x: torch tensor to be standardized
    :param mu_x: mean value
    :param sigma_x: standard deviation
    :param inverse: should we do the inverse transformation? Then please prove mu_x and sigma_x
    :return: 3-tuple w/ standardized x, the mean of x, std of x
    """
    
    if inverse:        
        return (x * sigma_x) + mu_x
    
    if mu_x is None:
        mu_x = torch.mean(x, dim=0)
    
    if sigma_x is None:
        sigma_x = torch.std(x, dim=0)
    
    x_standardize = (x - mu_x) / (sigma_x + 1e-8)
    
    return x_standardize, mu_x, sigma_x


def min_max_feature_scaling(x, min_x=None, range_x=None, inverse=False):
    """
    Transform the data to be in the interval [0, 1]
    
    :param x: array to be transformed
    :param min_x: min value of original x
    :param range_x: range of original x
    :param inverse: inverse transformation
    """
    
    if inverse:
        return x * range_x + min_x
    
    if min_x is None:
        min_x = torch.min(x, dim=0)[0]
    
    if range_x is None:
        range_x = torch.max(x, dim=0)[0] - min_x
    
    x_transformed = (x - min_x) / (range_x + 1e-8)
    
    return x_transformed, min_x, range_x
    
    
def minusone_one_feature_scaling(x, min_x=None, range_x=None, inverse=False):
    """
    Transform the data to be in the interval [-1, 1]

    :param x: array to be transformed
    :param min_x: min value of original x
    :param range_x: range of original x
    :param inverse: inverse transformation
    """

    if inverse:
        # return (2 * x + 1) * range_x/2 + min_x
        return (x + 1) / 2 * range_x + min_x

    if min_x is None:
        min_x = torch.min(x, dim=0)[0]
        
    if range_x is None:
        range_x = torch.max(x, dim=0)[0] - min_x

    x_transformed = 2 * (x - min_x) / (range_x + 1e-8) - 1

    return x_transformed, min_x, range_x  
