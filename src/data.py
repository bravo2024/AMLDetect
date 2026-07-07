
"""data.py - synthetic fallback and real dataset loader."""
from pathlib import Path
import numpy as np

FEATURES=["feat_%02d"%i for i in range(12)]

def make_synthetic(n=4000,seed=42):
    rng=np.random.default_rng(seed); d=len(FEATURES); X=rng.normal(size=(n,d))
    w=rng.normal(size=d)*(rng.random(d)<0.5); logits=X@w+0.6*X[:,0]*X[:,1]-1.4
    y=(rng.random(n)<1/(1+np.exp(-logits))).astype(int)
    return {"X":X,"y":y,"features":FEATURES}

def load_real_banknote():
    import pandas as pd
    import urllib.request
    
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/banknote_authentication.csv"
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_path = raw_dir / "banknote_authentication.csv"
    
    if not csv_path.exists():
        print(f"Downloading real dataset from {url} ...")
        urllib.request.urlretrieve(url, csv_path)
        
    df = pd.read_csv(csv_path, header=None, names=["variance", "skewness", "curtosis", "entropy", "class"])
    # The target column is 'class'
    target = "class"
    num = df.drop(columns=[target]).select_dtypes("number")
    return {"X":num.to_numpy(),"y":df[target].astype(int).to_numpy(),"features":list(num.columns)}

def load_real(csv_name,target):
    import pandas as pd; df=pd.read_csv(Path("data/raw")/csv_name)
    num=df.drop(columns=[target]).select_dtypes("number")
    return {"X":num.to_numpy(),"y":df[target].astype(int).to_numpy(),"features":list(num.columns)}

if __name__=="__main__":
    d=load_real_banknote(); print("Real X",d["X"].shape,"pos",int(d["y"].sum()))
