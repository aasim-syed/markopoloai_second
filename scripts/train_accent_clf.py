import json, numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from accent_features import extract_accent_features

rows=[json.loads(l) for l in open("data/processed/train_manifest.jsonl","r",encoding="utf-8")]
rows=[r for r in rows if r.get("label") in ("BD","UNK")]
X,y=[],[]
for r in rows:
    f=extract_accent_features(r["audio"])
    if f is not None:
        X.append(f); y.append(1 if r["label"]=="BD" else 0)
X=np.stack(X); y=np.array(y)
strat = y if (y.sum()!=0 and y.sum()!=len(y)) else None
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42,stratify=strat)
clf=RandomForestClassifier(n_estimators=400,random_state=42); clf.fit(Xtr,ytr)
acc=accuracy_score(yte, clf.predict(Xte))
print("Accent clf accuracy (BD vs UNK):", round(acc*100,2), "%")
