"""Підготовка датасету: ознаки текстур DTD (структури переплетення) -> data/dataset.csv.

Джерело: Describable Textures Dataset (Cimpoi et al., CVPR 2014, DOI 10.1109/CVPR.2014.461), копія на
Hugging Face Hub із зафіксованою ревізією. Класи: woven, knitted, braided, meshed (по 120 зображень).
Ознаки (104) – як у ПР-4/ПР-5: статистика яскравості (37), спектр Фур'є (39), LBP (28); S = 256 px.
Числа записуються з 8 значущими цифрами, щоб файл був однаковим на різних ОС.
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download
from PIL import Image
from skimage.feature import local_binary_pattern

REPO = 'cansa/Describable-Textures-Dataset-DTD'
REVISION = '96e70d392d5167ee33fcf7c251119411c9c8e820'   # зафіксована ревізія датасету
CLASSES = ['woven', 'knitted', 'braided', 'meshed']
SIZE = 256
OUT = Path('data/dataset.csv')


def load_gray(path, size=SIZE):
    im = Image.open(path).convert('L')
    m = min(im.size)
    left, top = (im.width - m) // 2, (im.height - m) // 2
    im = im.crop((left, top, left + m, top + m)).resize((size, size), Image.BILINEAR)
    return np.asarray(im, dtype=np.float64) / 255.0


def feat_base(g):
    h = np.histogram(g, bins=32, range=(0, 1))[0].astype(float)
    h /= h.sum()
    mu, sd = g.mean(), g.std() + 1e-8
    z = (g - mu) / sd
    return np.r_[h, mu, sd, (z ** 3).mean(), (z ** 4).mean(), -(h[h > 0] * np.log2(h[h > 0])).sum()]


def feat_fft(g, n_rad=24, n_ang=12):
    n = g.shape[0]
    win = np.outer(np.hanning(n), np.hanning(n))
    power = np.abs(np.fft.fftshift(np.fft.fft2((g - g.mean()) * win))) ** 2
    yy, xx = np.indices(power.shape) - n // 2
    r = np.hypot(xx, yy) / (n / 2)
    theta = np.mod(np.arctan2(yy, xx), np.pi)
    band = (r > 0.02) & (r <= 1.0)
    rad = np.histogram(r[band], bins=n_rad, range=(0.02, 1.0), weights=power[band])[0] / power[band].sum()
    mid = (r > 0.05) & (r < 0.6)
    ang = np.histogram(theta[mid], bins=n_ang, range=(0, np.pi), weights=power[mid])[0]
    ang /= ang.sum()
    pm = power[mid]
    return np.r_[np.log(rad + 1e-9), ang, np.log(pm.max() / pm.mean()), r[mid][pm.argmax()], ang.max() / ang.mean()]


def feat_lbp(g):
    g8 = (g * 255).astype(np.uint8)
    hists = []
    for p, rad in ((8, 1), (16, 2)):
        code = local_binary_pattern(g8, p, rad, method='uniform').astype(int)
        h = np.bincount(code.ravel(), minlength=p + 2)[:p + 2].astype(float)
        hists.append(h / h.sum())
    return np.concatenate(hists)


def main():
    root = snapshot_download(REPO, repo_type='dataset', revision=REVISION, local_dir='data/raw',
                             allow_patterns=[f'images/{c}/*.jpg' for c in CLASSES])
    paths = sorted(p for c in CLASSES for p in Path(root, 'images', c).glob('*.jpg'))
    rows = []
    for p in paths:
        g = load_gray(p)
        rows.append(np.r_[feat_base(g), feat_fft(g), feat_lbp(g)])
    df = pd.DataFrame(rows, columns=[f'f{i:03d}' for i in range(len(rows[0]))])
    df.insert(0, 'image', [p.name for p in paths])
    df['label'] = [p.parent.name for p in paths]
    OUT.parent.mkdir(exist_ok=True)
    df.to_csv(OUT, index=False, float_format='%.8g', lineterminator='\n')
    md5 = hashlib.md5(OUT.read_bytes()).hexdigest()
    print(f'{OUT}: {df.shape[0]} зображень, {df.shape[1] - 2} ознак, класи {df.label.value_counts().to_dict()}')
    print(f'MD5 {md5}')


if __name__ == '__main__':
    main()
