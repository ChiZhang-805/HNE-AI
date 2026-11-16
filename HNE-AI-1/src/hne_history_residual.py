"""One HNE rule: preserve historical risk and learn conditional corrections."""
import hashlib

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import SplineTransformer, StandardScaler

from .hne_additive import AdditiveWarning, person_weights


class HistoryResidualWarning:
    def __init__(self, history_C=.1, correction_C=.1, crossfit=True, seed=20261110):
        self.history_C, self.correction_C = history_C, correction_C
        self.crossfit, self.seed = crossfit, seed

    def _transform(self, x, fit=False):
        x = x[:, self.correction_columns]
        if fit:
            self.imputer = SimpleImputer(strategy='median', keep_empty_features=True)
            self.scaler = StandardScaler()
            self.basis = SplineTransformer(n_knots=3, degree=2, include_bias=False, extrapolation='linear')
            self.basis_scaler = StandardScaler()
            x = np.clip(self.scaler.fit_transform(self.imputer.fit_transform(x)), -8, 8)
            x = self.basis_scaler.fit_transform(self.basis.fit_transform(x))
        else:
            x = np.clip(self.scaler.transform(self.imputer.transform(x)), -8, 8)
            x = self.basis_scaler.transform(self.basis.transform(x))
        return x / self.penalty_scale

    def fit(self, x, y, people, groups):
        people, groups, y = np.asarray(people), np.asarray(groups), np.asarray(y)
        self.history_columns = np.flatnonzero(groups == 'H')
        self.correction_columns = np.flatnonzero(groups != 'H')
        h = x[:, self.history_columns]
        self.history = AdditiveWarning(C=self.history_C, groups=groups[self.history_columns]).fit(h, y, people)
        weights, multiplier = person_weights(people, y)
        if self.crossfit:
            ids = sorted(np.unique(people), key=lambda p: hashlib.sha256(f'{self.seed}:{p}'.encode()).hexdigest())
            allocation = {p: i % 3 for i, p in enumerate(ids)}
            fold = np.array([allocation[p] for p in people])
            history_probability = np.empty(len(y))
            for k in range(3):
                train, held = fold != k, fold == k
                history = AdditiveWarning(C=self.history_C, groups=groups[self.history_columns]).fit(
                    h[train], y[train], people[train])
                history_probability[held] = history.predict_proba(h[held])
        else:
            history_probability = self.history.predict_proba(h)
        offset = logit(np.clip(history_probability, 1e-7, 1-1e-7)) + np.log(multiplier)
        _, inverse, counts = np.unique(groups[self.correction_columns], return_inverse=True, return_counts=True)
        self.penalty_scale = np.repeat(np.sqrt(counts[inverse]*3), 3)
        z = self._transform(x, fit=True)

        def objective(theta):
            coefficient, intercept = theta[:-1], theta[-1]
            eta = offset + z @ coefficient + intercept
            loss = weights @ (np.logaddexp(0., eta)-y*eta) + .5/self.correction_C*(coefficient @ coefficient)
            residual = weights*(expit(eta)-y)
            gradient = np.r_[z.T @ residual + coefficient/self.correction_C, residual.sum()]
            return float(loss), gradient

        result = minimize(objective, np.zeros(z.shape[1]+1), method='L-BFGS-B', jac=True,
            options={'maxiter':2000, 'ftol':1e-11, 'gtol':1e-5, 'maxls':40})
        self.coef_, self.intercept_ = result.x[:-1], result.x[-1]
        self.optimization = dict(success=bool(result.success), iterations=int(result.nit),
            objective=float(result.fun), max_abs_gradient=float(abs(result.jac).max()))
        return self

    def predict_proba(self, x):
        history = self.history.predict_proba(x[:, self.history_columns])
        return expit(logit(np.clip(history, 1e-7, 1-1e-7)) + self._transform(x) @ self.coef_ + self.intercept_)
