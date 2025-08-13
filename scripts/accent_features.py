import librosa, numpy as np
def extract_accent_features(path):
    try: y,sr=librosa.load(path, sr=22050, mono=True)
    except: return None
    mfcc=librosa.feature.mfcc(y=y,sr=sr,n_mfcc=13); mf=np.hstack([mfcc.mean(1), mfcc.std(1)])
    sc=librosa.feature.spectral_centroid(y=y,sr=sr); scs=[sc.mean(), sc.std()]
    try: f0=librosa.yin(y, fmin=50, fmax=500); f0s=[np.nanmean(f0), np.nanstd(f0)]
    except: f0s=[0.0,0.0]
    return np.hstack([mf, scs, f0s]).astype(np.float32)
