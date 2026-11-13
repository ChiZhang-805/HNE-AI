"""A shared, regularized warning rule on personal-relative HNE coordinates."""
from __future__ import annotations

import numpy as np
from scipy.special import expit, logit
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.ensemble import HistGradientBoostingClassifier


def person_weights(person, y, domains=None):
    _, inverse, count = np.unique(person, return_inverse=True, return_counts=True)
    base = 1./count[inverse]
    if domains is not None:
        domains = np.asarray(domains)
        for domain in np.unique(domains):
            mask = domains == domain
            base[mask] /= base[mask].sum()
    positive_multiplier = base[y == 0].sum()/max(base[y == 1].sum(), 1e-8)
    weight = base * np.where(y == 1, positive_multiplier, 1.)
    return weight * len(weight)/weight.sum(), positive_multiplier


def validation_score(y, probability):
    if len(np.unique(y)) == 2:
        return float(roc_auc_score(y, probability))
    return -float(log_loss(y, probability, labels=[0,1]))


class AdditiveWarning:
    def __init__(self, C=.1, spline=True, groups=None):
        self.C, self.spline, self.groups = C, spline, groups

    def fit(self, x, y, person, domains=None):
        self.imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        self.scaler = StandardScaler()
        x = self.scaler.fit_transform(self.imputer.fit_transform(x))
        x = np.clip(x, -8, 8)
        self.basis = SplineTransformer(n_knots=3, degree=2, include_bias=False, extrapolation="linear") if self.spline else None
        if self.basis:
            x = self.basis.fit_transform(x)
        self.basis_scaler = StandardScaler()
        x = self.basis_scaler.fit_transform(x)
        if self.groups is not None:
            repeats = x.shape[1]//len(self.groups)
            _, inv, counts = np.unique(self.groups, return_inverse=True, return_counts=True)
            self.penalty_scale = np.repeat(np.sqrt(counts[inv]*repeats), repeats)
        else:
            self.penalty_scale = np.ones(x.shape[1])
        x = x/self.penalty_scale
        weight, self.positive_multiplier = person_weights(np.asarray(person), y, domains)
        self.model = LogisticRegression(C=self.C, max_iter=2000, solver="lbfgs")
        self.model.fit(x, y, sample_weight=weight)
        return self

    def predict_proba(self, x):
        x = np.clip(self.scaler.transform(self.imputer.transform(x)), -8, 8)
        if self.basis:
            x = self.basis.transform(x)
        x = self.basis_scaler.transform(x)/self.penalty_scale
        scores = self.model.decision_function(x)-np.log(self.positive_multiplier)
        return expit(scores)


class BoostedWarning:
    def __init__(self, C=.1, seed=20261110):
        self.C, self.seed = C, seed

    def fit(self, x, y, person):
        self.keep = np.array([len(np.unique(c[np.isfinite(c)])) > 1 for c in x.T])
        weight, self.positive_multiplier = person_weights(np.asarray(person), y)
        self.model = HistGradientBoostingClassifier(max_iter=150, max_leaf_nodes=7,
            l2_regularization=1/self.C, learning_rate=.05, random_state=self.seed)
        self.model.fit(x[:,self.keep], y, sample_weight=weight)
        return self

    def predict_proba(self, x):
        p = self.model.predict_proba(x[:,self.keep])[:,1]
        return expit(logit(np.clip(p,1e-7,1-1e-7))-np.log(self.positive_multiplier))


def history_features(x, scales=(24,72,168)):
    features = []
    for hours in scales:
        window = x[:, -hours:]
        hours=window.shape[1]
        valid = np.isfinite(window)
        first = np.argmax(valid, axis=1)
        last = hours-1-np.argmax(valid[:, ::-1], axis=1)
        row, col = np.indices(first.shape)
        delta = window[row, last, col]-window[row, first, col]
        values = [np.nanmean(window, axis=1), np.nanstd(window, axis=1),
                  np.nanquantile(window, .75, axis=1)-np.nanquantile(window, .25, axis=1),
                  delta, 1-valid.mean(1)]
        features.extend(values)
    return np.concatenate(features, axis=1)
