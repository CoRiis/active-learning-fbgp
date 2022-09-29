# metrics
import torch


def rmse(preds, targets):
    """
    Root Mean Square Error
    
    :param preds: torch.tensor
    :param targets: torch.tensor
    """
    return torch.sqrt(torch.mean(torch.pow(preds - targets, 2), dim=0))


def mse(preds, targets):
    """
    Mean Square Error

    :param preds: torch.tensor
    :param targets: torch.tensor
    """
    return torch.mean(torch.pow(preds - targets, 2), dim=0)


def mae(preds, targets):
    """
    Mean Absolute Error

    :param preds: torch.tensor
    :param targets: torch.tensor
    """
    return torch.mean(torch.abs(preds - targets), dim=0)


def rrse(preds, targets):
    """
    Root Relative Squared Error
    (https://www.gepsoft.com/GeneXproTools/AnalysesAndComputations/MeasuresOfFit/RootRelativeSquaredError.htm)
    """
    return torch.sqrt(mse(preds, targets) / mse(torch.mean(targets, dim=0), targets))


def rea(preds, targets):
    """
    Relative Absolute Error

    :param preds: torch.tensor
    :param targets: torch.tensor
    """
    return mae(preds, targets) / mae(torch.mean(targets), targets)


def picp(preds, ci_lower, ci_upper):
    """
    Prediction Interval Coverage Probability [1]

    [1] A. Khosravi, E. Mazloumi, S. Nahavandi, D. Creighton, and J. W. C. Van Lint,
    “Prediction intervals to account for uncertainties in travel time prediction,”
    IEEE Trans. Intell. Transp. Syst., vol. 12, no. 2, pp. 537–547, Jun. 2011.

    :param preds: torch.tensor
    :param ci_lower: torch.tensor, lower confidence interval for the corresponding targets
    :param ci_upper: torch.tensor, upper confidence interval for the corresponding targets
    """
    counter = 0
    n = preds.shape[0]
    for i in range(n):
        counter += int(ci_lower[i] < preds[i] < ci_upper[i])
    return counter / n
