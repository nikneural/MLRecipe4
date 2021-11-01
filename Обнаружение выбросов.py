import numpy as np
from sklearn.covariance import EllipticEnvelope
from sklearn.datasets import make_blobs

features, _ = make_blobs(n_samples=10,
                         n_features=2,
                         centers=1,
                         random_state=1)

# Заменить значение первого наблюдения предельными значениями
features[0, 0] = 10000
features[0, 1] = 10000

# Создать детектор
outlier_detector = EllipticEnvelope(contamination=.1)

outlier_detector.fit(features)

# Предсказать выбросы
print(outlier_detector.predict(features))

# Основным ограничением этого подхода является необходимость указания параметра загрязнения contamination,
# который представляет собой долю наблюдений, являющихся выбросами, — значение, которое мы не знаем

# Если мы ожидаем, что наши данные будут иметь несколько выбросов, мы можем задать параметр contamination с
# каким-нибудь небольшим значением. Однако, если мы считаем, что данные, скорее всего, будут иметь выбросы,
# мы можем установить для него более высокое значение.

# Вместо того чтобы смотреть на наблюдения в целом, мы можем взглянуть на
# отдельные признаки и идентифицировать в этих признаках предельные значения,
# используя межквартильный размах (МКР, IQR):

feature = features[:, 0]


def indicies_of_outliers(x):
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - (iqr * 1.5)
    upper_bound = q3 + (iqr * 1.5)
    return np.where((x > upper_bound) | (x < lower_bound))


print(indicies_of_outliers(feature))
