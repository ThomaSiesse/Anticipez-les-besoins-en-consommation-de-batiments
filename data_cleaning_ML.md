# Guide — Data Cleaning pour le Machine Learning

---

## 1. Explorer les données (EDA)

Avant de nettoyer quoi que ce soit, comprendre ce qu'on a.

```python
import pandas as pd

df = pd.read_csv("dataset.csv")

df.shape          # dimensions
df.dtypes         # types de colonnes
df.head()         # aperçu
df.describe()     # statistiques descriptives
df.info()         # résumé global
df.value_counts() # Différentes valeurs
```

---

## 2. Gérer les valeurs manquantes

### Détecter

```python
df.isnull().sum()
df.isnull().mean() * 100  # pourcentage par colonne
```

### Traiter

```python
# Supprimer les lignes avec trop de NaN
df.dropna(thresh=len(df.columns) * 0.5, inplace=True)

# Imputation numérique (moyenne / médiane)
df["colonne"].fillna(df["colonne"].median(), inplace=True)

# Imputation catégorielle (mode)
df["colonne"].fillna(df["colonne"].mode()[0], inplace=True)

# Imputation avancée (KNN, régression...)
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
df_imputed = imputer.fit_transform(df)
```

> **Règle générale :** < 5% de NaN → imputation simple. > 30% → envisager de supprimer la colonne.

---

## 3. Supprimer les doublons

```python
df.duplicated().sum()           # compter
df.drop_duplicates(inplace=True)
```

---

## 4. Corriger les types de données

```python
df["date"] = pd.to_datetime(df["date"])
df["prix"] = pd.to_numeric(df["prix"], errors="coerce")
df["categorie"] = df["categorie"].astype("category")
```

---

## 5. Nettoyer les valeurs textuelles

```python
df["nom"] = df["nom"].str.strip()        # espaces inutiles
df["nom"] = df["nom"].str.lower()        # harmoniser la casse
df["nom"] = df["nom"].str.replace(r"[^a-z0-9]", "", regex=True)  # caractères spéciaux
```

---

## 6. Détecter et traiter les outliers

### Méthode IQR (robuste)

```python
Q1 = df["colonne"].quantile(0.25)
Q3 = df["colonne"].quantile(0.75)
IQR = Q3 - Q1

df_clean = df[
    (df["colonne"] >= Q1 - 1.5 * IQR) &
    (df["colonne"] <= Q3 + 1.5 * IQR)
]
```

### Méthode Z-score

```python
from scipy import stats
df = df[(abs(stats.zscore(df["colonne"])) < 3)]
```

> **Attention :** ne pas supprimer systématiquement les outliers — ils peuvent être informatifs.

---

## 7. Encoder les variables catégorielles

```python
# One-Hot Encoding (variables sans ordre)
df = pd.get_dummies(df, columns=["couleur"], drop_first=True)

# Label Encoding (variables ordinales)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df["niveau"] = le.fit_transform(df["niveau"])

# Ordinal Encoding (ordre défini)
from sklearn.preprocessing import OrdinalEncoder
oe = OrdinalEncoder(categories=[["faible", "moyen", "élevé"]])
df[["niveau"]] = oe.fit_transform(df[["niveau"]])
```

---

## 8. Normaliser / Standardiser les features

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# StandardScaler : moyenne=0, écart-type=1 (SVM, régression logistique...)
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[numerical_cols])

# MinMaxScaler : valeurs entre [0, 1] (réseaux de neurones, KNN...)
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df[numerical_cols])
```

> ⚠️ Toujours `fit` sur le train set uniquement, puis `transform` sur le test set.

---

## 9. Gérer le déséquilibre de classes

```python
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Oversampling (SMOTE)
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

# Undersampling
rus = RandomUnderSampler(random_state=42)
X_res, y_res = rus.fit_resample(X, y)

# Vérifier la distribution
pd.Series(y).value_counts(normalize=True)
```

---

## 10. Vérification finale avant modélisation

```python
# Aucune valeur manquante
assert df.isnull().sum().sum() == 0

# Pas de doublons
assert df.duplicated().sum() == 0

# Types corrects
df.dtypes

# Distribution des features
df.describe()

# Corrélations
import seaborn as sns
sns.heatmap(df.corr(), annot=True, fmt=".2f")
```

---

## Récapitulatif des étapes

| Étape | Action                           |
| ----- | -------------------------------- |
| 1     | Explorer les données (EDA)       |
| 2     | Gérer les valeurs manquantes     |
| 3     | Supprimer les doublons           |
| 4     | Corriger les types               |
| 5     | Nettoyer les textes              |
| 6     | Traiter les outliers             |
| 7     | Encoder les catégorielles        |
| 8     | Normaliser / Standardiser        |
| 9     | Gérer le déséquilibre de classes |
| 10    | Vérification finale              |

---

_Fichier généré pour usage ML — Stack : Python / pandas / scikit-learn / imbalanced-learn_
