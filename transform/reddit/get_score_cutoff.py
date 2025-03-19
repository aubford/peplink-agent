# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares


def _log_score_cutoff_percentile():
    def print_cutoff_percent(scores: list[int]):
        print(
            f"Preserve Percentile: {round(get_score_cutoff_percentile(scores) * 100)}%"
        )

    print_cutoff_percent([1, 2, 2])
    print_cutoff_percent([1, 1, 1, 1])
    print_cutoff_percent([7, 15, 64])
    print_cutoff_percent([1, 1, 7, 1])

    print_cutoff_percent([1, 2, 3, 4, 5])
    print_cutoff_percent([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    print_cutoff_percent([1, 1, 1, 1, 1, 1, 1, 1, 3, 5])
    print_cutoff_percent([1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 5])
    print_cutoff_percent([1, 2, 3, 4, 5, 6, 7, 18, 27])
    print_cutoff_percent([1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 1, 1])
    print_cutoff_percent([66, 66, 66])
    print_cutoff_percent([1, 1, 1, 1, 1, 500000, 1, 1, 1, 1, 1])
    print_cutoff_percent([1, 500000])

    print_cutoff_percent([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    print_cutoff_percent([1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1])
    print_cutoff_percent([1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1])
    print_cutoff_percent([1, 1, 1, 1, 1, 1, 1, 1, 1, 7, 1])
    print_cutoff_percent([1, 1, 1, 1, 1, 1, 1, 1, 1, 50, 1, 1, 1])
    print_cutoff_percent([50, 50, 50, 50, 1, 50, 50, 50, 50, 50])


def _log_distribution_factor(cv: float):
    cv_test_dist = np.linspace(0, 3)
    plt.plot(
        cv_test_dist,
        _get_distribution_factor(cv_test_dist),
        label="Distribution Factor",
    )
    plt.legend()

    print(f"get_distribution_factor(0): {round(_get_distribution_factor(0), 3)}")
    print(f"get_distribution_factor(1): {round(_get_distribution_factor(1), 3)}")
    print(f"get_distribution_factor(2): {round(_get_distribution_factor(2), 3)}")
    print(f"get_distribution_factor(3): {round(_get_distribution_factor(3), 3)}")


def _log_engagement_norm():
    plt.plot(np.linspace(0, 100, 1000), _get_engagement_norm(np.linspace(0, 100, 1000)))
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.minorticks_on()
    plt.grid(True, which="minor", linestyle=":", linewidth=0.2)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.legend()

    print(f"get_engagement_norm(0): {_get_engagement_norm(0)}")
    print(f"get_engagement_norm(1): {_get_engagement_norm(1)}")
    print(f"get_engagement_norm(2): {_get_engagement_norm(2)}")
    print(f"get_engagement_norm(3): {_get_engagement_norm(3)}")
    print(f"get_engagement_norm(4): {_get_engagement_norm(4)}")
    print(f"get_engagement_norm(5): {_get_engagement_norm(5)}")
    print(f"get_engagement_norm(6): {_get_engagement_norm(6)}")
    print(f"get_engagement_norm(7): {_get_engagement_norm(7)}")
    print(f"get_engagement_norm(8): {_get_engagement_norm(8)}")
    print(f"get_engagement_norm(9): {_get_engagement_norm(9)}")
    print(f"get_engagement_norm(10): {_get_engagement_norm(10)}")
    print(f"get_engagement_norm(20): {_get_engagement_norm(20)}")
    print(f"get_engagement_norm(30): {_get_engagement_norm(30)}")
    print(f"get_engagement_norm(100): {_get_engagement_norm(100)}")
    print(f"get_engagement_norm(1000): {_get_engagement_norm(1000)}")
    print(f"get_engagement_norm(100000): {_get_engagement_norm(10000000)}")


# space = np.linspace(0, 100)
# plt.plot(space, get_engagement_norm(space), label="engagement_norm")
# plt.legend()
# plt.show()

# # %%
# def _target_equations(params: np.ndarray) -> np.ndarray:
#     B, M, nu = params
#     def f(x: float) -> float:
#         return 1.0 / (1.0 + np.exp(-B*(x - M)))**(1.0/nu)
#     return np.array([
#         f(0) - 0.01,
#         f(1) - 0.75,
#         f(3) - 0.99,
#     ])

# # Numerically solve for B, M, nu
# initial_guess = np.array([1.0, 1.0, 1.0])
# solution = root(_target_equations, initial_guess).x
# print(f"solution: {solution}")
# B_fit, M_fit, nu_fit = solution


def _logistic_function(x: float, B: float, M: float, nu: float) -> float:
    """Generalized logistic function with parameters B (steepness), M (midpoint), and nu (asymmetry)"""
    return 1.0 / (1.0 + np.exp(-B * (x - M))) ** (1.0 / nu)


def _get_distribution_factor(cv) -> float:
    """
    Single logistic-type formula satisfying:
      f(0) ≈ 0.01,
      f(1) ≈ 0.75,
      f(2) ≈ 0.90,
      f(3) ≈ 0.99,
    rising gently from 1 -> 3.

    Use SciPy least_squares to fit the parameters.
    """

    def _target_equations(params: np.ndarray) -> np.ndarray:
        B, M, nu = params
        return np.array(
            [
                _logistic_function(0, B, M, nu) - 0.01,
                _logistic_function(1, B, M, nu) - 0.75,
                _logistic_function(2, B, M, nu) - 0.90,
                _logistic_function(3, B, M, nu) - 0.99,
            ]
        )

    initial_guess = np.array([2.4, -3.2, 0.0001])
    solution = least_squares(_target_equations, initial_guess, method="lm").x
    B_fit, M_fit, nu_fit = solution

    return _logistic_function(cv, B_fit, M_fit, nu_fit) - 0.025


def _stretched_exponential(
    x: float, a: float, p: float, log_base: float = 2.2
) -> float:
    """Satisfies f(a)=0.5 and f(0)=0"""
    k = (np.log(log_base) / (a**p)) ** (1 / p)
    return 1 - np.exp(-((k * x) ** p))


def _get_engagement_norm(engagement) -> float:
    """
    Engagement normalization to 0-1 range
    It should satisfy these constraints for engagement values:
    f(0)=0
    f(5)≈0.5
    f(100)≈0.95
    """
    # k = -1/5 * np.log(0.5)
    # return 1 - np.exp(-k * engagement)
    return _stretched_exponential(engagement, 4.1, 0.52, 2)


def _scale_distribution_factor(dist: float, p: float = 0.5, k: float = 2.0) -> float:
    """
    Scale the distribution factor by polynomial and power functions.
    """
    poly = k * dist - np.power(dist, k)
    return (np.power(dist, p) + poly) / 2


def get_score_cutoff_percentile(scores: list[int]) -> float:
    """
    Calculate a cutoff percentile based on the score distribution of the comments.

    In a low engagement environment, we don't have enough data to apply a score-based heuristic so we should return
    closer to 100% of comments.
    If score distribution is narrow, we should be less strict with our score requirements.
    If there is high engagement and a wide score distribution, we should be very strict without our score requirements
    with max selectivity being only the 20% best comments.
    """
    scores_array = np.array(scores)
    engagement = np.sum(scores_array - 1)  # Total upvotes (excluding initial 1 point)
    mean = np.mean(scores_array)
    std_dev = np.std(scores_array)
    cv = std_dev / (mean + 1e-8)

    engagement_norm = _get_engagement_norm(engagement)

    distribution_factor = _get_distribution_factor(cv)
    scaled_distribution_factor = _scale_distribution_factor(distribution_factor)

    cutoff_percentile = engagement_norm * scaled_distribution_factor
    return 1 - cutoff_percentile
