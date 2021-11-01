# Первый способ, отбросить выбросы

import pandas as pd

houses = pd.DataFrame()
houses['Цена'] = [534433, 392333, 293222, 4322032]
houses['Ванные'] = [2, 3.5, 2, 116]
houses['Кв_футы'] = [1500, 2500, 1500, 48000]

print(houses[houses['Ванные'] < 20])

# Второй способ пометить их как выбросы и включить их в качестве признака

import numpy as np

houses['Выброс'] = np.where(houses['Ванные'] < 20, 0, 1)
print(houses)

# Третий способ, преобразовать признак, чтобы ослабить эффект выброса

houses["Логарифм_кв_футов"] = [np.log(x) for x in houses['Кв_футы']]
print(houses)
