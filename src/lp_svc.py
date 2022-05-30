"""Experiment utilities for minimum Lp-margin Support Vector Classifier."""

from typing import Any, Dict, Tuple, Union

import cvxpy as cp
import numpy as np
from cvxpy.expressions.variable import Variable
from cvxpy.problems.problem import Problem
from sklearn.metrics import accuracy_score

from src.data_utils.preprocessing import train_val_splits_classification_tabular


def solve_svc_problem(
    x: np.ndarray, y: np.ndarray, p: float = 1.0, solver: Union[None, str] = "MOSEK"
) -> Tuple[float, Problem, Variable]:
    """Solve the linear Support Vector Classifier problem
    argmin ||w||_p s.t. for all i: yi <xi, w> >= 1
    where ||.||_p is the L_p norm.
    """
    n, d = x.shape  # Number of data points and data dimension.

    w = cp.Variable(d)
    objective = cp.Minimize(cp.norm(w, p=p))
    constraints = [cp.multiply(y.squeeze(), x @ w) >= 1]
    prob = cp.Problem(objective, constraints)

    results = prob.solve(solver=solver)

    return results, prob, w


def predict(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    y_pred = np.inner(x, w)
    assert y_pred.shape == (x.shape[0],)

    y_pred = np.sign(y_pred)
    y_pred[y_pred == 0] = 1  # Arbitrarily assign zeros.
    return y_pred


def svc_experiment(
    dataset: str,
    p: float = 1.0,
    label_noise: float = 0.0,
    solver: Union[None, str] = "MOSEK",
    n_splits: int = 10,
    random_split_size: float = 0.0,
    normalize_data: bool = True,
    random_state: Union[None, int] = None,
) -> Dict[str, Any]:
    cross_val_res = {"status": [], "w": [], "val_acc": []}
    for x_train, y_train, x_val, y_val in train_val_splits_classification_tabular(
        dataset, label_noise, normalize_data, n_splits, random_split_size, random_state
    ):
        # Run optimization and compute test accuracy.
        result, prob, w = solve_svc_problem(x_train, y_train, p=p, solver=solver)
        w = w.value
        y_pred = predict(x_val, w)
        acc = accuracy_score(y_val, y_pred)

        cross_val_res["status"].append(prob.status)
        cross_val_res["w"].append(w)
        cross_val_res["val_acc"].append(acc)

    return cross_val_res
