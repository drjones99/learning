import numpy as np
from scipy.special import gamma
from functools import lru_cache

@lru_cache(maxsize=256)
def weibull_count_pmf(x, lambda_, c):
    """
    Probability mass function for the Weibull count distribution.
    """
    def alpha(x, j):
        if x == 0:
            return gamma(c*j + 1) / gamma(j + 1)
        else:
            return sum(alpha(x-1, m) * gamma(c*j - c*m + 1) / gamma(j - m + 1) for m in range(x, j))

    return sum((-1)**(x+j) * (lambda_**j) * alpha(x, j) / gamma(c*j + 1) for j in range(x, 50))  # Truncate at 50

def frank_copula(u, v, kappa):
    """
    Frank copula function.
    """
    return (-1/kappa)*np.log(1+(np.exp(-kappa*u)-1)*(np.exp(-kappa*v)-1)/(np.exp(-kappa)-1+1e-8))  # Add a small constant to avoid division by zero

def bivariate_pmf(x, y, lambda_1, lambda_2, c, kappa):
    """
    Bivariate probability mass function using Weibull count marginals and Frank copula.
    """
    F1 = sum(weibull_count_pmf(i, lambda_1, c) for i in range(x))
    F2 = sum(weibull_count_pmf(i, lambda_2, c) for i in range(y))
    F1_next = F1 + weibull_count_pmf(x, lambda_1, c)
    F2_next = F2 + weibull_count_pmf(y, lambda_2, c)
    
    return max(0, (frank_copula(F1_next, F2_next, kappa) - 
                  frank_copula(F1_next, F2, kappa) -
                  frank_copula(F1, F2_next, kappa) +
                  frank_copula(F1, F2, kappa)))

def predict_score(team_a_goals, team_b_goals, c=1.05, kappa=-0.456):
    """
    Predict the score distribution given the expected goals for each team.
    """
    max_goals = 6
    score_probs = np.zeros((max_goals+1, max_goals+1))
    
    for i in range(max_goals+1):
        for j in range(max_goals+1):
            score_probs[i, j] = bivariate_pmf(i, j, team_a_goals, team_b_goals, c, kappa)
    
    # Normalize the score probabilities to ensure they add up to 1
    score_probs /= np.sum(score_probs)
    
    return score_probs

# Example usage
team_a_goals = 2.8 # Expected goals for team A
team_b_goals = 0.65 # Expected goals for team B

score_distribution = predict_score(team_a_goals, team_b_goals)

print("Probability distribution of scores:")
print("     0     1     2     3     4     5     6")
for i, row in enumerate(score_distribution):
    print(f"{i:2} {row[0]:.3f} {row[1]:.3f} {row[2]:.3f} {row[3]:.3f} {row[4]:.3f} {row[5]:.3f} {row[6]:.3f}")  

# Calculate probabilities for 1X2 market
team_a_win_prob = np.sum(np.tril(score_distribution, -1))
draw_prob = np.sum(np.diag(score_distribution))
team_b_win_prob = np.sum(np.triu(score_distribution, 1))

print(f"\n1X2 Market Probabilities:")
print(f"Team A Win: {team_a_win_prob:.4f}")
print(f"Draw: {draw_prob:.4f}")
print(f"Team B Win: {team_b_win_prob:.4f}")

# Calculate probabilities for Over/Under 2.5 goals
under_2_5_prob = np.sum(score_distribution[:3, :3])
over_2_5_prob = 1 - under_2_5_prob

print(f"\nOver/Under 2.5 Goals Probabilities:")
print(f"Under 2.5: {under_2_5_prob:.4f}")
print(f"Over 2.5: {over_2_5_prob:.4f}")