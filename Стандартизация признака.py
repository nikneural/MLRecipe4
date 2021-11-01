import numpy as np
from sklearn import preprocessing

x = np.array([[-1000.1],
              [-200.2],
              [500.5],
              [600.6],
              [9000.9]])

scaler = preprocessing.StandardScaler()
standardized = scaler.fit_transform(x)

print(standardized)

print("Среднее:", round(standardized.mean()))
print("Стандартное отклонение:", standardized.std())


# Если наши данные имеют значительные выбросы, это может негативно повлиять на стандартизацию,
# сказываясь на среднем значении и дисперсии признака

robust_scaler = preprocessing.RobustScaler()
print(robust_scaler.fit_transform(x))
