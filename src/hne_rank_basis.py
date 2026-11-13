"""Scale-only comparison: the same HNE additive rule on training ranks."""
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import QuantileTransformer, SplineTransformer, StandardScaler

from .hne_additive import AdditiveWarning, person_weights


class RankBasisWarning(AdditiveWarning):
    def fit(self, x, y, person, domains=None):
        self.imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        self.scaler = QuantileTransformer(n_quantiles=min(1001, len(x)),
                                         output_distribution="uniform", subsample=None,
                                         random_state=20261110)
        x = self.scaler.fit_transform(self.imputer.fit_transform(x))
        self.basis = SplineTransformer(n_knots=3, degree=2, include_bias=False,
                                      extrapolation="linear")
        x = self.basis.fit_transform(x)
        self.basis_scaler = StandardScaler()
        x = self.basis_scaler.fit_transform(x)
        repeats = x.shape[1]//len(self.groups)
        _, inv, counts = np.unique(self.groups, return_inverse=True, return_counts=True)
        self.penalty_scale = np.repeat(np.sqrt(counts[inv]*repeats), repeats)
        weight, self.positive_multiplier = person_weights(np.asarray(person), y, domains)
        self.model = LogisticRegression(C=self.C, max_iter=2000, solver="lbfgs")
        self.model.fit(x/self.penalty_scale, y, sample_weight=weight)
        return self
