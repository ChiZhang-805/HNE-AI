"""Personal HNE updates from fully matured, same-person feedback."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit


def source_coordinates(source, x):
    z = np.clip(source.scaler.transform(source.imputer.transform(x)), -8, 8)
    if source.basis is not None:
        z = source.basis.transform(z)
    return source.basis_scaler.transform(z) / source.penalty_scale


def update_parameters(z, y, prior, C, condition):
    def objective(theta):
        eta = z @ theta
        delta = theta-prior
        loss = np.sum(np.logaddexp(0., eta)-y*eta) + .5/C*(delta @ delta)
        gradient = z.T @ (expit(eta)-y) + delta/C
        return float(loss), gradient
    bounds = [(None,None)]*len(prior)
    if condition == 'affine':
        bounds[1] = (0.,None)
    result = minimize(objective, prior.copy(), method='L-BFGS-B', jac=True, bounds=bounds,
        options={'maxiter':500,'ftol':1e-10,'gtol':1e-5})
    return result.x


def prequential(source, x, y, people, days, condition, C, config):
    people, days, y = np.asarray(people), np.asarray(days), np.asarray(y)
    source_probability = source.predict_proba(x)
    source_logit = logit(np.clip(source_probability, 1e-7, 1-1e-7))
    if condition == 'coefficients':
        z = np.column_stack([source_coordinates(source,x), np.ones(len(x))])
        prior = np.r_[source.model.coef_[0], source.model.intercept_[0]-np.log(source.positive_multiplier)]
    elif condition == 'affine':
        z = np.column_stack([np.ones(len(x)),source_logit])
        prior = np.array([0.,1.])
    else:
        z = np.ones((len(x),1)); prior = np.zeros(1)
    predictions = source_probability.copy()
    counts = np.zeros(len(x),dtype=int)
    latest = np.full(len(x),-1,dtype=int)
    if condition == 'frozen':return predictions,counts,latest
    for person in np.unique(people):
        indices = np.flatnonzero(people==person)
        indices = indices[np.argsort(days[indices],kind='stable')]
        parameters = prior.copy(); last_update = -100000
        used_count, used_latest = 0,-1
        for index in indices:
            d=days[index]
            available=indices[(days[indices]+config['feedback_maturity_days']<=d) &
                              (days[indices]>=d-config['memory_calendar_days'])]
            if len(available)>=config['minimum_mature_origins'] and d-last_update>=config['update_calendar_days']:
                if condition == 'intercept':
                    offset=source_logit[available]
                    labels=y[available]
                    def objective(theta):
                        eta=offset+theta[0]
                        return float(np.sum(np.logaddexp(0.,eta)-labels*eta)+.5*theta[0]**2/C), np.array([(expit(eta)-labels).sum()+theta[0]/C])
                    parameters=minimize(objective,np.zeros(1),jac=True,method='L-BFGS-B').x
                else:
                    parameters=update_parameters(z[available],y[available],prior,C,condition)
                last_update=d; used_count=len(available); used_latest=int(days[available].max())
            eta=float(z[index]@parameters)
            if condition=='intercept':eta+=source_logit[index]
            predictions[index]=expit(eta)
            counts[index],latest[index]=used_count,used_latest
    return predictions,counts,latest
