"""Finite HNE candidate comparison on the two already-exposed development cohorts."""
from pathlib import Path
import argparse
import json
import sys
import time
import warnings
import pickle

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.hne_additive import AdditiveWarning, BoostedWarning, person_weights, validation_score, history_features
from src.hne_coordinates import coordinate_mask
from threadpoolctl import threadpool_limits


def metrics(y, p):
    return dict(n=len(y), positives=int(y.sum()), auroc=float(roc_auc_score(y,p)) if len(np.unique(y)) == 2 else None,
                ap=float(average_precision_score(y,p)) if y.sum() else None, brier=float(brier_score_loss(y,p)))


def run(models, folds, feature_folder=None, output_folder=None, datasets=None, scales=(24,72,168)):
    config = json.loads((ROOT/"configs/development.json").read_text())
    feature_folder=Path(feature_folder) if feature_folder else ROOT/'data/features'
    output_folder=Path(output_folder) if output_folder else ROOT/'results/development'
    for dataset in (datasets or config["datasets"]):
        z = np.load(feature_folder/f"{dataset}.npz")
        meta = pd.read_csv(feature_folder/f"{dataset}.csv", dtype={"person_id":str})
        y, persons, partition = z["y"], meta.person_id.to_numpy(), z["fold"]
        for fold in folds:
            train = ~np.isin(partition, [fold, (fold+1)%5])
            val, test = partition == (fold+1)%5, partition == fold
            old = np.load(ROOT/f"data/development/{dataset}/fold{fold}.npz") if 'windows' not in z else None
            for model in models:
                destination = output_folder/f"{dataset}/{model}/fold{fold}"
                if (destination/"metrics.json").exists():
                    continue
                start = time.time()
                destination.mkdir(parents=True, exist_ok=True)
                groups = z["groups"].copy()
                if model.startswith("History") or model == 'Missingness-LR':
                    windows=[z['windows'][mask] for mask in (train,val,test)] if 'windows' in z else [old[f'{s}_x'] for s in ('train','validation','test')]
                    if model == 'Missingness-LR':
                        xtrain, xval, xtest = [np.concatenate([np.isfinite(x[:,-s:]).mean(1) for s in scales],axis=1) for x in windows]
                    else:
                        xtrain, xval, xtest = [history_features(x,scales=scales) for x in windows]
                    # Stored fold arrays and metadata are in the same global sample order.
                    group = None
                else:
                    keep = coordinate_mask(model)
                    if model == "HNE-no-network":
                        keep = ~np.isin(groups, ["N","NE"])
                    elif model == "HNE-no-environment":
                        keep = ~np.isin(groups, ["E","NE"])
                    elif model == "HNE-history":
                        keep = groups == "H"
                    xtrain, xval, xtest = [z["pooled"][mask][:,keep] for mask in (train,val,test)]
                    group = groups[keep]
                trials, best, best_score = [], None, -np.inf
                if model == "HNE-set":
                    from src.hne_set import fit_predict
                    for decay in config["mlp_weight_decay"]:
                        result = fit_predict(z["nodes"][train], z["pooled"][train], y[train], persons[train],
                            z["nodes"][val], z["pooled"][val], y[val], persons[val],
                            z["nodes"][test], z["pooled"][test], groups,
                            seed=config["seeds"][0], weight_decay=decay, output=destination/f"decay{decay}")
                        score = validation_score(y[val], result["val_probability"])
                        trials.append(dict(weight_decay=decay, validation=score))
                        if score > best_score:
                            best_score, best, chosen = score, result, dict(weight_decay=decay)
                    vp, tp = best["val_probability"], best["test_probability"]
                else:
                    for C in config["ridge_C"]:
                        if model.endswith("HGB"):
                            fitted = BoostedWarning(C=C,seed=config["seeds"][0]).fit(xtrain,y[train],persons[train])
                            vpred = fitted.predict_proba(xval)
                        else:
                            fitted = AdditiveWarning(C=C, spline=model.startswith("HNE"),groups=group).fit(xtrain,y[train],persons[train])
                            vpred = fitted.predict_proba(xval)
                        score = validation_score(y[val],vpred)
                        trials.append(dict(C=C, validation=score))
                        if score > best_score:
                            best_score,best,chosen,vp = score,fitted,dict(C=C),vpred
                    tp = best.predict_proba(xtest)
                    with open(destination/"model.pkl","wb") as file:
                        pickle.dump(best,file)
                info = dict(dataset=dataset,model=model,fold=fold,stage="development",chosen=chosen,
                            trials=trials,validation=metrics(y[val],vp),held_person=metrics(y[test],tp),seconds=time.time()-start)
                (destination/"metrics.json").write_text(json.dumps(info,indent=2))
                for mask,prediction,split in ((val,vp,"validation"),(test,tp,"held_person")):
                    frame=meta[mask].copy()
                    frame["probability"]=prediction
                    frame.to_csv(destination/f"{split}.csv",index=False)
                print(json.dumps(info),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--models",nargs="+",default=["HNE-additive","History-LR","History-HGB","HNE-HGB"])
    parser.add_argument("--folds",nargs="+",type=int,default=list(range(5)))
    parser.add_argument('--features',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--datasets',nargs='+')
    parser.add_argument('--scales',nargs=3,type=int,default=[24,72,168])
    args=parser.parse_args()
    warnings.filterwarnings("ignore",message="All-NaN slice encountered")
    warnings.filterwarnings("ignore",message="Mean of empty slice")
    warnings.filterwarnings("ignore",message="Degrees of freedom <= 0 for slice")
    with threadpool_limits(limits=2):
        run(args.models,args.folds,args.features,args.output,args.datasets,args.scales)
