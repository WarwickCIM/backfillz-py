from __future__ import annotations
from dataclasses import dataclass
import pickle

import stan
from stan.fit import Fit
from stan.model import Model


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


@dataclass
class Stan:
    """Bundle a Stan fit with its model."""
    model: Model
    fit: Fit

    def save(self, file: str) -> None:
        with open(file + ".pkl", "wb") as f:
            pickle.dump({'model': self.model, 'fit': self.fit}, f, protocol=-1)

    @staticmethod
    def load(file: str) -> Stan:
        with open(file + ".pkl", "rb") as f:
            saved = pickle.load(f)
            return Stan(saved.model, saved.fit)

    def equal(self, other: Stan) -> bool:
        return str(self) == str(other)

def generate_fit() -> Stan:
    """Make PyStan fit from the Eight Schools example."""
    model = stan.build(schools_code, data=schools_data, random_seed=1)
    return Stan(model, model.sample(num_chains=4, num_samples=1000))


if __name__ == '__main__':
    generate_fit()
