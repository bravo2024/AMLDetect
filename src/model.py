
"""model.py - classification; LightGBM if installed else NumPy logistic baseline."""
import numpy as np
from src.core import Standardizer,LogisticRegression,train_test_split,roc_auc_score,accuracy_score,f1_score
PREDICT_KIND="tabular"
def _estimator():
    try:
        import lightgbm as lgb
        return "lightgbm",lgb.LGBMClassifier(n_estimators=150,learning_rate=0.05,max_depth=4,
                                             num_leaves=15,min_child_samples=50,verbose=-1)
    except Exception:
        return "numpy-logreg",LogisticRegression(lr=0.3,epochs=500)
def fit_and_evaluate(data):
    X=np.asarray(data["X"],float); y=np.asarray(data["y"],int)
    Xtr,Xte,ytr,yte=train_test_split(X,y,0.25,7); sc=Standardizer().fit(Xtr)
    name,est=_estimator(); est.fit(sc.transform(Xtr),ytr)
    proba=est.predict_proba(sc.transform(Xte))[:,1] if name=="lightgbm" else est.predict_proba(sc.transform(Xte))
    # threshold at a 1% alert budget: an AML desk reviews a queue, not a 0.5 cutoff
    thresh=float(np.quantile(proba,0.99)); pred=(proba>=thresh).astype(int)
    alerted=pred==1
    metrics={"backend":name,"n_train":int(len(Xtr)),"n_test":int(len(Xte)),"roc_auc":roc_auc_score(yte,proba),
             "alert_rate":float(pred.mean()),"alert_precision":float(yte[alerted].mean()) if alerted.any() else 0.0,
             "alert_recall":float(pred[yte==1].mean()) if (yte==1).any() else 0.0,
             "f1":f1_score(yte,pred),"positive_rate":float(yte.mean())}
    return {"scaler":sc,"estimator":est,"backend":name,"features":data.get("features")},metrics
def predict_proba(model,X):
    Xs=model["scaler"].transform(np.asarray(X,float)); est=model["estimator"]
    return est.predict_proba(Xs)[:,1] if model["backend"]=="lightgbm" else est.predict_proba(Xs)
