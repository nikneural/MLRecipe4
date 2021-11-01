# В ваших данных имеются пропущенные значения, и требуется заполнить или предсказать их значения.

import numpy as np
from fancyimpute import KNN
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs

features, _ = make_blobs(n_samples=1000,
                         n_features=2,
                         random_state=1)

# Импутация (imputation) — процесс замещения пропущенных, некорректных или несостоятельных
# значений другими значениями

scaler = StandardScaler()
standardized_features = scaler.fit_transform(features)

# Заменить первое значение первого признака на пропущенное значение
true_value = standardized_features[0, 0]
standardized_features[0, 0] = np.nan

# Предсказать пропущенные значения в матрице признаков
features_knn_inputed = KNN(k=5, verbose=0).complete(standardized_features)

print("Истинное значение:", true_value)
print("Импутированное значение:", features_knn_inputed[0,0])

