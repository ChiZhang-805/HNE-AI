"""Public tree estimators used in the CPU development comparison."""
import numpy as np


def tree_scores(method, seed, train, label, val):
    if method == 'XGBoost':
        from xgboost import XGBClassifier
        model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=.05, n_jobs=1,
                              device='cpu', tree_method='exact', random_state=seed)
    elif method == 'LightGBM':
        from lightgbm import LGBMClassifier
        model = LGBMClassifier(n_estimators=100, num_leaves=15, min_child_samples=5, learning_rate=.05,
                               n_jobs=1, verbosity=-1, force_col_wise=True, random_state=seed)
    elif method == 'Random Forest':
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=200, min_samples_leaf=2, n_jobs=1, random_state=seed)
    else:
        raise ValueError(method)
    train, val = np.ascontiguousarray(train), np.ascontiguousarray(val)
    if method == 'LightGBM':
        import pandas as pd
        columns = [f'f{i}' for i in range(train.shape[1])]
        train, val = pd.DataFrame(train, columns=columns), pd.DataFrame(val, columns=columns)
    model.fit(train, label)
    return model.predict_proba(val)[:, 1]
