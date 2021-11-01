import numpy as np
from sklearn.preprocessing import PolynomialFeatures

features = np.array([[2, 3],
                     [2, 3],
                     [2, 3]])

polynomial_interaction = PolynomialFeatures(degree=2, include_bias=False)

print(polynomial_interaction.fit_transform(features))

# В реализованном классе PolynomialFeatures, есть признаки взаимодействия:
# Мы можем ограничить создаваемые признаки только признаками взаимодействия
interaction = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
print(interaction.fit_transform(features))