import json, csv, os
root="data/processed"
for split in ["train","valid"]:
    in_path=os.path.join(root, f"{split}_manifest.jsonl")
    out_path=os.path.join(root, f"{split}_metadata.csv")
    if not os.path.exists(in_path): continue
    with open(in_path,"r",encoding="utf-8") as f, open(out_path,"w",newline="",encoding="utf-8") as g:
        w=csv.writer(g, delimiter="|")
        for line in f:
            r=json.loads(line)
            w.writerow([r["audio"], r["text"], "spk0", "bn"])
    print("Wrote", out_path)
