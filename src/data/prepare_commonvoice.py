import argparse, json
from datasets import load_dataset
from ..utils.logging import ensure_dir

"""Load Common Voice Bengali and save manifest with locale-based BD filtering when available."""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='train')
    ap.add_argument('--out', default='data/processed/commonvoice_bn.jsonl')
    ap.add_argument('--only_bd', action='store_true')
    args = ap.parse_args()

    ds = load_dataset('mozilla-foundation/common_voice_11_0', 'bn', split=args.split)
    with open(args.out, 'w', encoding='utf-8') as w:
        for ex in ds:
            # Common Voice has speaker metadata; use 'locale' or 'accent' if present
            locale = ex.get('locale') or ex.get('accent') or ''
            if args.only_bd and 'BD' not in str(locale).upper():
                continue
            if ex.get('path') and ex.get('sentence'):
                item = {
                    'audio': ex['path'],
                    'text': ex['sentence'],
                    'sr': 48000,  # CV default; will resample later
                    'meta': {'locale': locale, 'client_id': ex.get('client_id', '')}
                }
                w.write(json.dumps(item, ensure_ascii=False) + "\n")
    print('Wrote manifest to', args.out)

if __name__ == '__main__':
    main()