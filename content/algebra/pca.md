# Principal Component Analysis (PCA)

PCA is een techniek voor dimensiereductie die vaak gebruikt wordt bij data-analyse en machine learning.

## Doel

- Reduceer de dimensionaliteit van data
- Behoud zoveel mogelijk variantie
- Vind de belangrijkste "richtingen" in de data

## Stappen

1. **Centreren** - Trek het gemiddelde af van de data
2. **Covariantiematrix** - Bereken de covariantiematrix
3. **Eigenwaarden/vectoren** - Bereken eigenwaarden en eigenvectoren
4. **Selecteren** - Kies de $k$ grootste eigenwaarden
5. **Transformeren** - Projecteer de data op de geselecteerde eigenvectoren

## Wiskundige basis

De covariantiematrix van gecentreerde data $X$ is:

$$
C = \frac{1}{n-1} X^T X
$$

De principale componenten zijn de eigenvectoren van $C$.

## Voorbeeld in Python

```python
from sklearn.decomposition import PCA
import numpy as np

# Voorbeeld data
X = np.random.rand(100, 5)

# PCA met 2 componenten
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

print("Oorspronkelijke vorm:", X.shape)
print("Gereduceerde vorm:", X_reduced.shape)
print("Verklaarde variantie:", pca.explained_variance_ratio_)
```
