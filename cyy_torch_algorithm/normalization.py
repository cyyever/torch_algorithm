import numpy as np
import numpy.typing as npt


def normalize(v: npt.NDArray) -> npt.NDArray:
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def row_norms(mat: npt.NDArray) -> npt.NDArray:
    return np.linalg.norm(mat, 2, axis=1)


def column_norms(mat: npt.NDArray) -> npt.NDArray:
    return np.linalg.norm(mat, 2, axis=0)


def normalize_for_heatmap(mat: npt.NDArray) -> npt.NDArray:
    rn = row_norms(mat) ** 0.5
    rn[rn == 0] = 1.0
    rn = 1 / rn
    cn = column_norms(mat) ** 0.5
    cn[cn == 0] = 1.0
    cn = 1 / cn
    mat = np.matmul(np.diag(rn), mat)
    mat = np.matmul(mat, np.diag(cn))
    return mat
