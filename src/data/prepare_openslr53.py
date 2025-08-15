import os, json, argparse, tarfile, glob
from urllib.request import urlretrieve
from tqdm import tqdm
from ..utils.logging import ensure_dir
from ..utils.audio import load_audio
from .features import extract_accent_features

OPENSLR_URL = "https://www.openslr.org/resources/53/"
FILES = [
    "dev.tar.gz", "test.tar.gz", "train.tar.gz"
]


def download_dataset(out_dir: str):
    ensure_dir(out_dir)
    for f in FILES:
        url = OPENSLR_URL + f
        dest = os.path.join(out_dir, f)
        if not os.path.exists(dest):
            print("Downloading", url)
            urlretrieve(url, dest)
        with tarfile.open(dest) as tar:
            tar.extractall(out_dir)


def create_manifest(root: str, out_jsonl: str, sr: int = 22050, limit: int = None):
    # Assumes extracted Kaldi-like structure with wav + text
    # Walk for .wav and matching transcript
    pairs = []
    for wav in glob.glob(os.path.join(root, "**", "*.wav"), recursive=True):
        txt = os.path.splitext(wav)[0] + ".txt"
        if os.path.exists(txt):
            with open(txt, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            pairs.append((wav, text))
            if limit and len(pairs) >= limit:
                break
    with open(out_jsonl, 'w', encoding='utf-8') as w:
        for wav, text in pairs:
            item = {"audio": wav, "text": text, "sr": sr}
            w.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Wrote {len(pairs)} items -> {out_jsonl}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='data/raw/openslr53')
    ap.add_argument('--manifest', default='data/processed/openslr53_all.jsonl')
    ap.add_argument('--limit', type=int, default=None)
    args = ap.parse_args()

    download_dataset(args.out)
    # Create a simple manifest; curation done later
    create_manifest(args.out, args.manifest, limit=args.limit)

if __name__ == '__main__':
    main()