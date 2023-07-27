from scipy.stats import pearsonr
import numpy as np


def RMSE(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    return np.sqrt(mse)


def R2_Score(y_true, y_pred):
    corr_matrix = np.corrcoef(y_true, y_pred)
    corr = corr_matrix[0, 1]
    R2 = corr ** 2

    return R2


def PCC(y_true, y_pred):
    corr, _ = pearsonr(y_true, y_pred)
    return corr


def evaluate(y_true, y_pred):
    rmse = RMSE(y_true, y_pred)
    r2 = R2_Score(y_true, y_pred)
    pcc = PCC(y_true, y_pred)

    return rmse, r2, pcc


if __name__ == '__main__':
    y = np.asarray([10, 20, 30, 40, 50])
    y_hat = np.asarray([11, 21, 32, 41, 51])

    print(RMSE(y, y_hat))
    print(R2_Score(y, y_hat))
    print(PCC(y, y_hat))
