# Требуется прошкалировать значения признаков в наблюдениях для получения единичной нормы (общей длиной 1).

import numpy as np
from sklearn.preprocessing import Normalizer

features = np.array([[0.5, 0.5],
                     [1.1, 3.4],
                     [1.5, 20.2],
                     [1.63, 34.4],
                     [10.9, 3.3]])

normalizer = Normalizer(norm='l2') # Евклидовая норма
print(normalizer.transform(features))

features_l1_norm = Normalizer(norm='l1').transform(features) # манхэттенская норма
print(features_l1_norm)
