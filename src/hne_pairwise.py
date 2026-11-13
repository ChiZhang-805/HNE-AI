"""The shared HNE spline head with an additional within-person ranking loss."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from .hne_additive import AdditiveWarning,person_weights


class PairwiseWarning(AdditiveWarning):
    def __init__(self,C=.1,alpha=.25,groups=None):
        super().__init__(C=C,spline=True,groups=groups)
        self.alpha=alpha

    def fit(self,x,y,person,domains=None):
        super().fit(x,y,person,domains)
        transformed=np.clip(self.scaler.transform(self.imputer.transform(x)),-8,8)
        transformed=self.basis_scaler.transform(self.basis.transform(transformed))/self.penalty_scale
        person=np.asarray(person);y=np.asarray(y)
        weight,_=person_weights(person,y,domains)
        positives=[];negatives=[];masses=[]
        for individual in np.unique(person):
            pos=np.flatnonzero((person==individual)&(y==1))
            neg=np.flatnonzero((person==individual)&(y==0))
            if not len(pos) or not len(neg):continue
            positives.extend(np.repeat(pos,len(neg)))
            negatives.extend(np.tile(neg,len(pos)))
            masses.extend(np.full(len(pos)*len(neg),1/(len(pos)*len(neg))))
        pos=np.asarray(positives,int);neg=np.asarray(negatives,int);mass=np.asarray(masses,float)
        mass/=mass.sum()
        self.pair_count=len(mass)

        def objective(parameters):
            coef=parameters[:-1];scores=transformed@coef+parameters[-1]
            cost=np.mean(weight*(np.logaddexp(0,scores)-y*scores))+np.dot(coef,coef)/(2*self.C*len(y))
            gradient=weight*(expit(scores)-y)/len(y)
            differences=scores[pos]-scores[neg]
            cost+=self.alpha*np.dot(mass,np.logaddexp(0,-differences))
            pair_gradient=-self.alpha*mass*expit(-differences)
            gradient+=np.bincount(pos,weights=pair_gradient,minlength=len(y))-np.bincount(neg,weights=pair_gradient,minlength=len(y))
            return cost,np.r_[transformed.T@gradient+coef/(self.C*len(y)),gradient.sum()]

        initial=np.r_[self.model.coef_[0],self.model.intercept_]
        fitted=minimize(objective,initial,jac=True,method='L-BFGS-B',options={'maxiter':400,'ftol':1e-10,'gtol':1e-6})
        self.model.coef_=fitted.x[:-1][None];self.model.intercept_=fitted.x[-1:]
        self.optimization=dict(converged=bool(fitted.success),iterations=int(fitted.nit),objective=float(fitted.fun))
        return self
