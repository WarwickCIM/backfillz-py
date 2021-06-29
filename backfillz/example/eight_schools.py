from __future__ import annotations

import stan  # type: ignore

from backfillz.stan import Stan


schools_code = """
data {
  int<lower=0> J;         // number of schools
  real y[J];              // estimated treatment effects
  real<lower=0> sigma[J]; // standard error of effect estimates
}
parameters {
  real mu;                // population treatment effect
  real<lower=0> tau;      // standard deviation in treatment effects
  vector[J] eta;          // unscaled deviation from mu by school
}
transformed parameters {
  vector[J] theta = mu + tau * eta;        // school treatment effects
}
model {
  target += normal_lpdf(eta | 0, 1);       // prior log-density
  target += normal_lpdf(y | theta, sigma); // log-likelihood
}
"""

schools_data = {"J": 8,
                "y": [28, 8, -3, 7, -1, 1, 18, 12],
                "sigma": [15, 10, 16, 11, 9, 11, 10, 18]}


def generate_fit() -> Stan:
    """Make PyStan fit from the Eight Schools example."""
    model = stan.build(schools_code, data=schools_data, random_seed=1)
    return Stan(model, model.sample(num_chains=4, num_samples=12000))


if __name__ == '__main__':
    generate_fit()
