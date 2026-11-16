"""Source-domain weighted boosted-tree controls on shared HNE coordinates."""
import numpy as np
from scipy.special import expit, logit
from sklearn.ensemble import HistGradientBoostingClassifier
from .hne_additive import person_weights


class SharedCoordinateBoosting:
    def __init__(self, C=.1):
        self.C = C

    def fit(self, x, y, people, domains):
        weights, self.multiplier = person_weights(people, y, domains)
        self.keep = np.asarray([len(np.unique(c[np.isfinite(c)])) > 1 for c in x.T])
        self.model = HistGradientBoostingClassifier(max_iter=150, max_leaf_nodes=7,
            l2_regularization=1/self.C, learning_rate=.05, random_state=20261110)
        self.model.fit(x[:, self.keep], y, sample_weight=weights)
        return self

    def predict_proba(self, x):
        p = self.model.predict_proba(x[:, self.keep])[:, 1]
        return expit(logit(np.clip(p, 1e-7, 1-1e-7))-np.log(self.multiplier))
