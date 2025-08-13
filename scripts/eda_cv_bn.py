import json, os, random, soundfile as sf
from statistics import mean
root="data/processed"
def load_manifest(p): 
    return [json.loads(l) for l in open(p,"r",encoding="utf-8")] if os.path.exists(p) else []
tr = load_manifest(os.path.join(root,"train_manifest.jsonl"))
va = load_manifest(os.path.join(root,"valid_manifest.jsonl"))
def dur(w): 
    try: i=sf.info(w); return i.frames/(i.samplerate or 1)
    except: return 0.0
durs=[dur(r["audio"]) for r in tr]
print("Train:",len(tr)," Valid:",len(va))
if durs: print("Avg dur:",round(mean([d for d in durs if d>0]),2),"s  Min/Max:",round(min(durs),2),"/",round(max(durs),2),"s")
print("\nSample texts:")
for r in random.sample(tr, min(5,len(tr))): print("-", r["text"][:120])
