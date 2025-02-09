# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares


def print_cutoff_percent(scores: list[int]):
    print(f"Preserve Percentile: {round(get_score_cutoff_percentile(scores) * 100)}%")


def get_engagement_norm(engagement: int) -> float:
    """
    Engagement normalization within 0-1 range
    If engagement is 0, return 0
    Engagement normalization should ramp up quickly to about 0.5 when engagement values are
    in the range of 0-5 and then start to taper off.
    """
    # if engagement < 1:
    #     return 0

    return engagement / (engagement + 5)


plt.plot(np.linspace(0, 100), get_engagement_norm(np.linspace(0, 100)))
plt.show()

print(f"get_engagement_norm(0): {get_engagement_norm(0)}")
print(f"get_engagement_norm(1): {get_engagement_norm(1)}")
print(f"get_engagement_norm(2): {get_engagement_norm(2)}")
print(f"get_engagement_norm(3): {get_engagement_norm(3)}")
print(f"get_engagement_norm(4): {get_engagement_norm(4)}")
print(f"get_engagement_norm(5): {get_engagement_norm(5)}")
print(f"get_engagement_norm(6): {get_engagement_norm(6)}")
print(f"get_engagement_norm(7): {get_engagement_norm(7)}")
print(f"get_engagement_norm(8): {get_engagement_norm(8)}")
print(f"get_engagement_norm(9): {get_engagement_norm(9)}")
print(f"get_engagement_norm(10): {get_engagement_norm(10)}")
print(f"get_engagement_norm(20): {get_engagement_norm(20)}")
print(f"get_engagement_norm(30): {get_engagement_norm(30)}")
print(f"get_engagement_norm(100): {get_engagement_norm(100)}")
# %%
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


def get_distribution_factor(cv: float) -> float:
    """
    Single logistic-type formula satisfying:
      f(0) ≈ 0.01,
      f(1) ≈ 0.75,
      f(3) ≈ 0.99,
    rising gently from near 0 to near 1.
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


cv_test_dist = np.linspace(0, 3)
plt.plot(cv_test_dist, get_distribution_factor(cv_test_dist), label="Distribution Factor")
plt.legend()
plt.show()

print(f"get_distribution_factor(0): {round(get_distribution_factor(0), 3)}")
print(f"get_distribution_factor(1): {round(get_distribution_factor(1), 3)}")
print(f"get_distribution_factor(2): {round(get_distribution_factor(2), 3)}")
print(f"get_distribution_factor(3): {round(get_distribution_factor(3), 3)}")


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
    print(f"\n\nscores: {scores_array}")
    engagement = np.sum(scores_array - 1)  # Total upvotes (excluding initial 1 point)
    mean = np.mean(scores_array)
    std_dev = np.std(scores_array)
    # print(f"std_dev: {std_dev}")
    cv = std_dev / (mean + 1e-8)
    print(f"cv: {round(cv, 4)}")

    engagement_norm = get_engagement_norm(engagement)
    print(f"engagement_norm: {round(engagement_norm, 4)}")

    distribution_factor = get_distribution_factor(cv)
    print(f"distribution_factor: {round(distribution_factor, 4)}")

    e_d = engagement_norm * distribution_factor
    print(f"e_d: {round(e_d, 4)}")

    base_cutoff = 0.2
    max_cutoff = 1.0

    cutoff_percentile = max_cutoff - (max_cutoff - base_cutoff) * e_d
    return cutoff_percentile


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


# %%

space = np.linspace(1e-8, 1)
plt.plot(space, np.power(space, 1 / 5), label="1/5")
plt.plot(space, np.power(space, 1 / 4), label="1/4")
plt.plot(space, np.power(space, 1 / 3), label="1/3")
plt.plot(space, np.power(space, 1 / 2), label="1/2")
plt.legend()
plt.show()
