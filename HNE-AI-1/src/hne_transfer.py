"""A coefficient prior for the same additive HNE risk function and frozen basis."""
from __future__ import annotations

from copy import deepcopy

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from .hne_additive import person_weights


class SourcePriorWarning:
    """Frozen source basis, target logistic fit with a source or zero L2 center.

    Minimize sum_i weight_i * BCE(y_i, z_i w + b) + ||w-center||²/(2C).
    The source center is its raw-logit coefficient vector. The intercept is not
    penalized; source intercept inheritance affects initialization only. Target
    class balancing is removed from prediction logits with its own multiplier.
    """

    def __init__(self, source, C=.1, prior="source"):
        if prior not in {"source", "zero"}:
            raise ValueError("prior must be source or zero")
        if C <= 0:
            raise ValueError("C must be positive")
        self.source = deepcopy(source)
        self.C, self.prior = C, prior

    def transform(self, x):
        source = self.source
        x = np.clip(source.scaler.transform(source.imputer.transform(x)), -8, 8)
        if source.basis:
            x = source.basis.transform(x)
        return source.basis_scaler.transform(x) / source.penalty_scale

    def fit(self, x, y, person, domains=None):
        z = self.transform(x).astype(np.float64)
        y = np.asarray(y, dtype=np.float64)
        if len(np.unique(y)) != 2:
            raise ValueError("Target training subset must contain both classes")
        weight, self.positive_multiplier = person_weights(np.asarray(person), y, domains)
        source_coef = np.asarray(self.source.model.coef_[0], dtype=np.float64)
        center = source_coef.copy() if self.prior == "source" else np.zeros_like(source_coef)
        initial_intercept = float(self.source.model.intercept_[0]) if self.prior == "source" else 0.
        initial = np.r_[center, initial_intercept]

        def objective(theta):
            coefficient, intercept = theta[:-1], theta[-1]
            logits = z @ coefficient + intercept
            delta = coefficient - center
            loss = np.dot(weight, np.logaddexp(0., logits) - y * logits) + .5/self.C * np.dot(delta, delta)
            residual = weight * (expit(logits) - y)
            gradient = np.r_[z.T @ residual + delta/self.C, residual.sum()]
            return float(loss), gradient

        result = minimize(objective, initial, method="L-BFGS-B", jac=True,
                          options={"maxiter": 2000, "ftol": 1e-11, "gtol": 1e-5, "maxls": 40})
        if not np.isfinite(result.fun) or not np.isfinite(result.x).all():
            raise FloatingPointError("Nonfinite target coefficient-prior fit")
        self.coef_, self.intercept_ = result.x[:-1].copy(), float(result.x[-1])
        self.optimization_ = {"success": bool(result.success), "iterations": int(result.nit),
            "message": str(result.message), "objective": float(result.fun),
            "maximum_absolute_gradient": float(np.max(np.abs(result.jac))),
            "squared_distance_to_center": float(np.sum((self.coef_-center)**2)),
            "source_basis_dimensions": z.shape[1]}
        return self

    def predict_proba(self, x):
        logits = self.transform(x) @ self.coef_ + self.intercept_
        return expit(logits - np.log(self.positive_multiplier))
