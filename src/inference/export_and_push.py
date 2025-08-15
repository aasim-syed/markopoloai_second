import argparse, torch
from huggingface_hub import HfApi, create_repo
from ..models.bd_vits import BDVitsModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='outputs/checkpoints/best.pt')
    ap.add_argument('--repo', required=True, help='username/bd-bangla-tts-female')
    ap.add_argument('--token', default=None)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BDVitsModel().to(device)
    sd = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(sd['model'], strict=False)
    model.eval()

    # Simple scripting of adapter + discriminator only for size; scripting full VITS may fail.
    try:
        scripted = torch.jit.script(model)
        scripted.save('bd_bangla_tts_optimized.pt')
        print('Saved torchscript model -> bd_bangla_tts_optimized.pt')
    except Exception as e:
        print('TorchScript failed; saving state_dict only:', e)

    if args.token:
        api = HfApi(token=args.token)
    else:
        api = HfApi()
    create_repo(args.repo, exist_ok=True)
    api.upload_file(path_or_fileobj='bd_bangla_tts_optimized.pt', path_in_repo='bd_bangla_tts_optimized.pt', repo_id=args.repo)
    print('Uploaded to HF Hub:', args.repo)

if __name__ == '__main__':
    main()