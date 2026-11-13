"""Shared family risks with an exact Bernoulli count layer."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import SplineTransformer, StandardScaler


def two_or_more(q, eligible):
    """P(count>=2), and its derivative with respect to each family probability."""
    q = np.where(eligible, q, 0.)
    n, families = q.shape
    zero, one, multiple = np.ones(n), np.zeros(n), np.zeros(n)
    for j in range(families):
        multiple = multiple + one*q[:, j]
        one, zero = one*(1-q[:, j])+zero*q[:, j], zero*(1-q[:, j])
    derivative = np.zeros_like(q)
    for held in range(families):
        z, o = np.ones(n), np.zeros(n)
        for j in range(families):
            if j != held:
                o, z = o*(1-q[:, j])+z*q[:, j], z*(1-q[:, j])
        derivative[:, held] = o*eligible[:, held]
    return multiple, derivative


class CardinalityWarning:
    def __init__(self, C=.1, global_weight=1., groups=None, max_iter=400):
        self.C, self.global_weight, self.groups, self.max_iter = C, global_weight, groups, max_iter

    def _transform(self, x):
        flat = x.reshape(-1, x.shape[-1])
        z = np.clip(self.scaler.transform(self.imputer.transform(flat)), -8, 8)
        return self.basis_scaler.transform(self.basis.transform(z))/self.penalty_scale

    def fit(self, x, family_y, eligible, y, person):
        n, families, features = x.shape
        flat = x.reshape(-1, features)
        self.imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        self.scaler = StandardScaler()
        z = np.clip(self.scaler.fit_transform(self.imputer.fit_transform(flat[eligible.ravel()])), -8, 8)
        self.basis = SplineTransformer(n_knots=3, degree=2, include_bias=False, extrapolation="linear")
        z = self.basis.fit_transform(z)
        self.basis_scaler = StandardScaler().fit(z)
        repeats = z.shape[1]//features
        _, inverse, counts = np.unique(self.groups, return_inverse=True, return_counts=True)
        self.penalty_scale = np.repeat(np.sqrt(counts[inverse]*repeats), repeats)
        design = self._transform(x)
        _, inverse, counts = np.unique(person, return_inverse=True, return_counts=True)
        weight = 1./counts[inverse]
        weight /= weight.sum()
        known = (family_y >= 0) & eligible
        family_weight = known/np.maximum(known.sum(1, keepdims=True), 1)*weight[:, None]
        family_weight /= family_weight.sum()
        target = np.maximum(family_y, 0)
        initial = np.zeros(design.shape[1]+1)
        prevalence = np.sum(family_weight*target)
        initial[-1] = np.log(np.clip(prevalence, 1e-6, 1-1e-6)/np.clip(1-prevalence, 1e-6, 1))
        strength = 1/(self.C*n)

        def objective(parameters):
            logits = (design@parameters[:-1]+parameters[-1]).reshape(n, families)
            q = expit(logits)
            loss = np.sum(family_weight*(np.logaddexp(0, logits)-target*logits))
            gradient = family_weight*(q-target)
            probability, derivative = two_or_more(q, eligible)
            probability = np.clip(probability, 1e-12, 1-1e-12)
            loss += self.global_weight*np.sum(weight*(-y*np.log(probability)-(1-y)*np.log1p(-probability)))
            global_delta = self.global_weight*weight*(probability-y)/(probability*(1-probability))
            gradient += global_delta[:, None]*derivative*q*(1-q)
            loss += .5*strength*np.dot(parameters[:-1], parameters[:-1])
            flat_gradient = gradient.ravel()
            grad = np.r_[design.T@flat_gradient+strength*parameters[:-1], flat_gradient.sum()]
            return loss, grad

        result = minimize(objective, initial, method="L-BFGS-B", jac=True,
                          options={"maxiter": self.max_iter, "ftol": 1e-10, "gtol": 1e-7})
        self.parameters = result.x
        self.optimization = {"success": bool(result.success), "message": str(result.message),
                             "iterations": int(result.nit), "objective": float(result.fun),
                             "maximum_gradient": float(np.max(np.abs(result.jac)))}
        return self

    def predict(self, x, eligible):
        q = expit(self._transform(x)@self.parameters[:-1]+self.parameters[-1]).reshape(x.shape[:2])
        risk, _ = two_or_more(q, eligible)
        return risk, np.where(eligible, q, np.nan)
