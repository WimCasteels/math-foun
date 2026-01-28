# Onzekerheid & Intervalschatting

## Soorten onzekerheid

### Aleatoire onzekerheid
Inherente willekeur in de data (niet reduceerbaar).

### Epistemische onzekerheid
Onzekerheid door gebrek aan kennis (reduceerbaar met meer data).

## Betrouwbaarheidsintervallen

Een $(1-\alpha)$ betrouwbaarheidsinterval voor $\mu$:

$$
\bar{x} \pm z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}
$$

- $\bar{x}$: steekproefgemiddelde
- $z_{\alpha/2}$: kritieke waarde (1.96 voor 95%)
- $\sigma$: standaarddeviatie
- $n$: steekproefgrootte

### Interpretatie

"Als we dit experiment vaak herhalen, zal 95% van de berekende intervallen de echte waarde bevatten."

## Foutmarges in ML

### Train/Test split

Schat de generalisatiefout met een test set.

### Cross-validation

Robuustere schatting door meerdere splits.

### Bootstrap

Resample de data om de verdeling van een schatter te bepalen.

## Model evaluatie met onzekerheid

```python
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Cross-validation scores
model = RandomForestClassifier()
scores = cross_val_score(model, X, y, cv=5)

mean_score = np.mean(scores)
std_score = np.std(scores)

# 95% betrouwbaarheidsinterval
ci = 1.96 * std_score / np.sqrt(len(scores))

print(f"Accuracy: {mean_score:.3f} ± {ci:.3f}")
```

## Waarschijnlijkheidsuitvoer

Veel ML modellen geven probabiliteiten:

```python
# Voorspelde kansen
probs = model.predict_proba(X_test)

# Onzekere voorspellingen (kans dicht bij 0.5)
uncertain = np.abs(probs[:, 1] - 0.5) < 0.1
```
