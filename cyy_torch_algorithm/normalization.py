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
    rn = 1 / (row_norms(mat) ** 0.5)
    cn = 1 / (column_norms(mat) ** 0.5)
    mat = np.matmul(np.diag(rn), mat)
    mat = np.matmul(mat, np.diag(cn))
    return mat
