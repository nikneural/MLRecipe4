import numpy as np
from sklearn.preprocessing import FunctionTransformer

features = np.array([[2, 3],
                     [2, 3],
                     [2, 3]])

def add_ten(x):
    return x + 10

ten_transformer = FunctionTransformer(add_ten)

print(ten_transformer.transform(features))
