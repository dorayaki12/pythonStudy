import numpy as np

t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]


# ２乗和誤差
def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)


# 交差エントロピー誤差
def cross_entropy_error(y, t):
    delta = 1e-7
    return np.sum(t * np.log(y + delta))


# 正解が近いバージョン
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
print(mean_squared_error(np.array(y), np.array(t)))
print(cross_entropy_error(np.array(y), np.array(t)))

# 正解が遠いバージョン
y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
print(mean_squared_error(np.array(y), np.array(t)))
print(cross_entropy_error(np.array(y), np.array(t)))
