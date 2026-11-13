from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.summarize_development import bootstrap_aucs,fold_auc

folder=ROOT/'results/development/summary'
out=ROOT/'results/development/figures';out.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'Times New Roman','font.size':13,'axes.spines.top':False,'axes.spines.right':False,
                     'pdf.fonttype':42,'ps.fonttype':42})
names=['DNB','FCM','CATCH','previous-History-LR','HNE-set-ensemble']
labels=['DNB','FCM','CATCH','History-LR','HNE']
for dataset in ['RESILIENT','BrisT1D-Open']:
    frame=pd.read_csv(folder/f'{dataset}_predictions.csv',dtype={'person_id':str})
    frame=frame[frame.primary_common_subset].copy()
    people,inv=np.unique(frame.person_id,return_inverse=True)
    weights=np.random.default_rng(20261110).multinomial(len(people),np.full(len(people),1/len(people)),size=2000)
    values=np.array([fold_auc(frame,n) for n in names])
    ci=np.array([np.nanquantile(bootstrap_aucs(frame,n,inv,weights),[.025,.975]) for n in names])
    fig,ax=plt.subplots(figsize=(7.0,4.0),layout='constrained')
    positions=np.arange(len(names));ax.bar(positions,values,width=.62,color=['.72']*4+['black'])
    ax.errorbar(positions,values,yerr=np.maximum(np.c_[values-ci[:,0],ci[:,1]-values].T,0),fmt='none',color='black',capsize=4,linewidth=1)
    for x,y,upper in zip(positions,values,ci[:,1]):ax.text(x,upper+.025,f'{y:.3f}',ha='center',va='bottom',fontsize=13)
    ax.set_xticks(positions,labels);ax.set_ylim(0,1.13);ax.set_yticks([0,.25,.5,.75,1]);ax.set_ylabel('AUROC')
    ax.set_title(dataset,fontsize=17,pad=14)
    fig.savefig(out/f'{dataset}.pdf');fig.savefig(out/f'{dataset}.png',dpi=220);plt.close(fig)
print(out)
