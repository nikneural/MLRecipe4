# Требуется удалить наблюдения, содержащие пропущенные значения.

import numpy as np

features = np.array([[1.1, 11.1],
                     [2.2, 22.2],
                     [3.3, 33.3],
                     [4.4, 44.4],
                     [np.nan, 55]])

# Оставить только те наблюдения, которые не (помечены ~) пропущены
print(features[~np.isnan(features).any(axis=1)])

# В качестве альтернативы можно удалить наблюдения, содержащие пропущенные
# значения, с помощью библиотеки pandas

import pandas as pd

dataframe = pd.DataFrame(features, columns=['признак_1', 'признак_2'])
print(dataframe.dropna())