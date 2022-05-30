"""Experiment utilities for minimum Lp margin Support Vector Regressor."""

from typing import Any, Dict, Optional, Tuple, Union

import cvxpy as cp
import numpy as np
from cvxpy.error import SolverError
from cvxpy.expressions.variable import Variable
from cvxpy.problems.problem import Problem
from sklearn.metrics import mean_squared_error

from src.data_utils.preprocessing import train_val_splits_regression_tabular


def solve_svr_problem(
    x: np.ndarray, y: np.ndarray, p: float = 1.0, solver: Union[None, str] = "MOSEK"
) -> Tuple[float, Problem, Variable, Variable]:
    """Solve the linear Support Vector Regressor problem
    argmin ||w||_p s.t. for all i: <xi, w> == yi
    where ||.||_p is the L_p norm.
    """
    n, d = x.shape  # Number of data points and data dimension.
    w = cp.Variable(d)
    b = cp.Variable(1)
    objective = cp.Minimize(cp.norm(w, p=p))
    constraints = [x @ w + b == y.squeeze()]
    prob = cp.Problem(objective, constraints)

    results = prob.solve(solver=solver)

    return results, prob, w, b


def predict(
    x: np.ndarray, w: np.ndarray, b: Optional[Union[np.ndarray, float]] = 0.0
) -> np.ndarray:
    return np.inner(x, w) + b


def svr_experiment(
    dataset: str,
    p: float = 1.0,
    noise: float = 0.0,
    solver: Union[None, str] = "MOSEK",
    normalize_data: bool = True,
) -> Dict[str, Any]:
    cross_val_res = {"status": [], "w": [], "val_mse": []}
    for x_train, y_train, x_val, y_val in train_val_splits_regression_tabular(
        dataset,
        noise,
        normalize_data,
    ):
        try:
            result, prob, w, b = solve_svr_problem(x_train, y_train, p=p, solver=solver)
        except SolverError:
            continue

        if prob.status == "optimal":
            w = w.value
            b = b.value
            y_pred = predict(x_val, w, b)
            mse = mean_squared_error(y_val, y_pred)

            cross_val_res["status"].append(prob.status)
            cross_val_res["w"].append(w)
            cross_val_res["val_mse"].append(mse)

    return cross_val_res
