
"""
Aqui hay 6 formas distintas de imputar missings
"""
# de apuntes de minería
import sklearn.impute as skl_imp
from sklearn.experimental import enable_iterative_imputer
# Moda: Solo nominales
imputer_moda = skl_imp.SimpleImputer(strategy='most_frequent', missing_values=np.nan)
# knn: Solo numéricas
imputer_knn = skl_imp.KNNImputer(n_neighbors=3)
# Chain equations: solo numéricas
imputer_itImp = skl_imp.IterativeImputer(max_iter=10, random_state=0)

import feature_engine.imputation as fe_imp
# Aleatoria: numéricas y nominales
imputer_rand = fe_imp.RandomSampleImputer()
# Mediana: solo numéricas
imputer_median = fe_imp.MeanMedianImputer(imputation_method='median')
# Media: solo nominales
imputer_mean = fe_imp.MeanMedianImputer(imputation_method='mean')