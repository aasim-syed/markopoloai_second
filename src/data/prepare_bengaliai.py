import argparse, json
from datasets import load_dataset

"""Load Bengali.AI train set tiny (as text prompts) where applicable for TTS text augmentation."""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='data/processed/bengaliai_texts.jsonl')
    args = ap.parse_args()
    ds = load_dataset('thesven/bengali-ai-train-set-tiny', split='train')
    with open(args.out, 'w', encoding='utf-8') as w:
        for ex in ds:
            text = str(ex.get('label', ''))
            if text:
                w.write(json.dumps({'text': text}) + "\n")
    print('Wrote', args.out)

if __name__ == '__main__':
    main()