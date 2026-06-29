"""

You ran two recommender systems on different users and got a score per user. 
You want to know: is the difference in average scores real, or just noise?

"""
from scipy.stats import t as t_dist
import numpy as np


def welchs_t(group_a, group_b, alpha=0.05):
    a = np.array(group_a, dtype=float)
    b = np.array(group_b, dtype=float)

    n_a, n_b = len(a), len(b) 
    mean_a, mean_b = a.mean(), b.mean()

    # Sample variances (ddof=1 = Bessel's correction)
    var_a = a.var(ddof=1)
    var_b = b.var(ddof=1)

    obs_diff = mean_b - mean_a

    std_error_a = var_a / n_a # std error of the mean
    std_error_b = var_b / n_b
    std_error_diff = np.sqrt(std_error_a + std_error_b)

    t_stat = obs_diff / std_error_diff

    # Degrees of freedom (Welch–Satterthwaite approximation)
    df = (var_a/n_a + var_b/n_b)**2 / (
        (var_a/n_a)**2 / (n_a - 1) +
        (var_b/n_b)**2 / (n_b - 1)
    )

    # two-tailed test
    p_value = (1 - t_dist.cdf(abs(t_stat), df)) * 2

    t_critical = t_dist.ppf(1 - alpha/2, df) # icdf

    ci_low  = obs_diff - t_critical * std_error_diff
    ci_high = obs_diff + t_critical * std_error_diff

    if p_value < alpha and obs_diff > 0:
        winner = "B is better"
    elif p_value < alpha and obs_diff < 0:
        winner = "A is better"
    else:
        winner = "inconclusive"

    return {
        "winner": winner,
        "ci": [ci_low, ci_high]
    }



group_a = np.random.normal(loc=0.35, scale=0.10, size=100) # change to 0.38
group_b = np.random.normal(loc=0.40, scale=0.10, size=100)

res = welchs_t(group_a, group_b)

print(res['winner'])