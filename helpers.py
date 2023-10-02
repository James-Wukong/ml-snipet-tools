from scipy import stats
from math import sqrt
import pandas as pd

def t_score(q, n):
    """
    get t_score.

    Parameters
    ----------
    q : float, The critical level to use
    n : sample number, between [2, 31]
    Returns
    -------
    t_score : float
    """
    # To find the T critical value
    return stats.t.ppf(q, df=n-1)

def p_value(t_score, n=10):
    """
    get p-value.

    Parameters
    ----------
    t_score : t_score, float
    n : sample number, between [2, 31]
    Returns
    -------
    p_value : float
    """
    # Determine the p-value, df=n-1
    return stats.t.sf(abs(t_score), df=n-1)

def pred_interval(points, m, b, x_0):
    """
    Calculating a prediction interval of an event

    Parameters
    ----------
    points : tuple, data points of linear algebra
    m : coefficient, float
    b : intercept, float
    x_0 : float, predicted based on this value
    Returns
    -------
    pred_lower_bound, pre_upper_bound
    """
    n = len(points)
    x_mean = sum(p.x for p in points) / n
    t_value = stats.t(n - 2).ppf(.975)
    standard_error = sqrt(sum((p.y - (m * p.x + b)) ** 2 for p in points) / (n - 2))
    margin_of_error = t_value * standard_error * \
                        sqrt(1 + (1 / n) + (n * (x_0 - x_mean) ** 2) / \
                        (n * sum(p.x ** 2 for p in points) - sum(p.x for p in points) ** 2))
    predicted_y = m*x_0 + b
    #Calculate prediction interval
    print(predicted_y - margin_of_error, predicted_y + margin_of_error)