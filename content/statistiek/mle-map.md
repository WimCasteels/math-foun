# Maximum Likelihood & MAP

## Maximum Likelihood Estimation (MLE)

Vind de parameters $\theta$ die de data het meest waarschijnlijk maken:

$$
\hat{\theta}_{MLE} = \arg\max_\theta P(D | \theta)
$$

### Log-likelihood

In de praktijk maximaliseren we de log-likelihood (makkelijker):

$$
\hat{\theta}_{MLE} = \arg\max_\theta \log P(D | \theta)
$$

### Voorbeeld: Normale verdeling

Gegeven data $x_1, \ldots, x_n$:

$$
\hat{\mu}_{MLE} = \frac{1}{n} \sum_{i=1}^n x_i
$$

$$
\hat{\sigma}^2_{MLE} = \frac{1}{n} \sum_{i=1}^n (x_i - \hat{\mu})^2
$$

## Maximum A Posteriori (MAP)

Combineer de likelihood met een **prior** over de parameters:

$$
\hat{\theta}_{MAP} = \arg\max_\theta P(\theta | D) = \arg\max_\theta P(D | \theta) \cdot P(\theta)
$$

### Verschil MLE vs MAP

| | MLE | MAP |
|---|---|---|
| Prior | Geen | Ja |
| Bij weinig data | Kan overfitten | Regulariseert |
| Formule | $\max P(D|\theta)$ | $\max P(D|\theta) P(\theta)$ |

## Relatie met Regularisatie

- Gaussische prior op $\theta$ → L2 regularisatie (Ridge)
- Laplace prior op $\theta$ → L1 regularisatie (Lasso)

## Python

```python
import numpy as np
from scipy.optimize import minimize
from scipy import stats

# Data
data = np.random.normal(5, 2, 100)

# MLE voor normale verdeling
mu_mle = np.mean(data)
sigma_mle = np.std(data)

print(f"MLE: mu = {mu_mle:.2f}, sigma = {sigma_mle:.2f}")

# Log-likelihood functie
def neg_log_likelihood(params, data):
    mu, sigma = params
    return -np.sum(stats.norm.logpdf(data, mu, sigma))

# Optimaliseren
result = minimize(neg_log_likelihood, [0, 1], args=(data,))
print(f"Optimized: mu = {result.x[0]:.2f}, sigma = {result.x[1]:.2f}")
```
