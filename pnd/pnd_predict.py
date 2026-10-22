r"""
pnd_predict.py - run the PANDA 1st-place (team PND) ensemble on your slides, on CPU.

Model and method: Kentaro Yoshioka, Yukihiro Nakamura (fam_taro), Arutema and
team PND, https://github.com/kentaroy47/Kaggle-PANDA-1st-place-solution
(CC-BY-NC 4.0: research and other non-commercial use only, credit the authors).

The ensemble, reproduced from their submitted_notebook.ipynb:

  part 1  EfficientNet-b1, folds 1, 4, 5: 64 tiles of 192 px from level 1,
          laid out 8x8 into one 1536 px image, 10 outputs (first 5 ordinal)
  part 2  EfficientNet-b0, folds 0-4: 36 tiles of 256 px, 6x6 into 1536 px,
          5 ordinal outputs
  TTA     every model sees 12 flips/rotations, and each slide is tiled twice
          (tile grid offset 0 and offset half a tile)
  blend   0.5 * part 1 + 0.5 * part 2, rounded to an ISUP grade 0-5

Usage (from your prostate folder, histo-env active):

    python pnd_predict.py slides --pnd-dir Kaggle-PANDA-1st-place-solution
    python pnd_predict.py slides\0005f7aaab2800f6170c399693a96917.tiff --pnd-dir ...
    python pnd_predict.py slides --pnd-dir ... --holdout --train-csv train.csv

--holdout is for PANDA train slides: the models were trained on almost all of
them, so it uses only the models that never saw each slide (and says which).

Results go to tiles\pnd_slides.csv; slides already in it are skipped, so a
long run can be stopped and restarted.

Research use only: this is not a validated diagnostic tool.
"""
import argparse
import csv
import glob
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

B1_FOLDS = [1, 4, 5]
B0_FOLDS = [0, 1, 2, 3, 4]
B1_FILE = "final_2_efficientnet-b1_kfold_{}_latest.pt"
B0_FILE = "efficientnet-b0famlabelsmodelsub_avgpool_tile36_imsize256_mixup_final_epoch20_fold{}.pth"
B1_TRAIN_CSV = "input/train-5kfold_remove_noisy_by_0622_rad_13_08_ka_15_10.csv"
B0_TRAIN_CSV = "output/model/final_1/local_preds_final_1_efficientnet-b1_removed_noise_thresh_16.csv"
HEADER = ["image_id", "pnd_isup", "pnd_score", "part_b1", "part_b0", "models",
          "isup_grade", "gleason_score", "data_provider", "seconds"]


# --------------------------------------------------------------------------- models

def efficientnet(name):
    from efficientnet_pytorch import EfficientNet
    # image_size=None gives TF "same" padding computed from the real input size,
    # which is what the PANDA-era efficientnet_pytorch did for 1536 px inputs.
    net = EfficientNet.from_name(name, image_size=None)
    net.set_swish(memory_efficient=False)
    return net


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p),
                            (x.size(-2), x.size(-1))).pow(1.0 / self.p)


class SEBlock(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        self.linear_1 = nn.Linear(ch, ch // r)
        self.linear_2 = nn.Linear(ch // r, ch)

    def forward(self, x):
        return x * torch.sigmoid(self.linear_2(F.relu(self.linear_1(x))))


class B1(nn.Module):
    """Part 1 (fam_taro): EfficientNet-b1, GeM pooling, SE head, 10 outputs."""

    def __init__(self):
        super().__init__()
        self.net = efficientnet("efficientnet-b1")
        ch = self.net._fc.in_features
        self.net._avg_pooling = GeM()
        self.net._fc = nn.Sequential(nn.Flatten(), SEBlock(ch), nn.Dropout(), nn.Linear(ch, 10))

    def forward(self, x):
        return self.net(x)[:, :5]


class B0(nn.Module):
    """Part 2 (arutema): EfficientNet-b0, average pooling, 5 outputs."""

    def __init__(self):
        super().__init__()
        self.basemodel = efficientnet("efficientnet-b0")
        self.myfc = nn.Linear(self.basemodel._fc.in_features, 5)
        self.basemodel._fc = nn.Identity()

    def forward(self, x):
        return self.myfc(self.basemodel(x))


def load_models(pnd_dir):
    weights = Path(pnd_dir) / "final_models"
    b1, b0 = {}, {}
    for k in B1_FOLDS:
        m = B1()
        ckpt = torch.load(weights / B1_FILE.format(k), map_location="cpu", weights_only=False)
        m.load_state_dict({key[4:]: v for key, v in ckpt["state_dict"].items()})  # drop "net."
        b1[k] = m.eval()
    for f in B0_FOLDS:
        m = B0()
        m.load_state_dict(torch.load(weights / B0_FILE.format(f), map_location="cpu"))
        b0[f] = m.eval()
    return b1, b0


# --------------------------------------------------------------------------- tiles

def read_level1(path):
    """Level 1 (4x downsampled) of a PANDA tiff as an RGB uint8 array."""
    try:
        import tifffile
        with tifffile.TiffFile(path) as tif:
            img = tif.pages[1].asarray()
    except ImportError:
        import openslide
        s = openslide.OpenSlide(str(path))
        rgba = np.asarray(s.read_region((0, 0), 1, s.level_dimensions[1]))
        img = np.where(rgba[..., 3:] == 0, 255, rgba[..., :3]).astype(np.uint8)
    return np.ascontiguousarray(img[..., :3])


def get_tiles(img, size, n, mode):
    """The n least-white size x size tiles; mode 2 shifts the grid by half a tile."""
    h, w, _ = img.shape
    pad_h = (size - h % size) % size + (size * mode) // 2
    pad_w = (size - w % size) % size + (size * mode) // 2
    img = np.pad(img, [[pad_h // 2, pad_h - pad_h // 2], [pad_w // 2, pad_w - pad_w // 2], [0, 0]],
                 constant_values=255)
    tiles = img.reshape(img.shape[0] // size, size, img.shape[1] // size, size, 3)
    tiles = tiles.transpose(0, 2, 1, 3, 4).reshape(-1, size, size, 3)
    if len(tiles) < n:
        tiles = np.pad(tiles, [[0, n - len(tiles)], [0, 0], [0, 0], [0, 0]], constant_values=255)
    order = np.argsort(tiles.reshape(len(tiles), -1).sum(-1), kind="stable")[:n]
    return tiles[order]


def mosaic(tiles, column_major):
    """Lay n tiles into a sqrt(n) square, inverted (tissue bright) and scaled to 0-1."""
    g = int(round(len(tiles) ** 0.5))
    s = tiles.shape[1]
    grid = tiles.reshape(g, g, s, s, 3)
    if column_major:  # part 1 used cv2.hconcat of vconcat'ed rows
        grid = grid.transpose(1, 0, 2, 3, 4)
    image = grid.transpose(0, 2, 1, 3, 4).reshape(g * s, g * s, 3)
    image = (255 - image.astype(np.float32)) / 255.0
    return torch.from_numpy(image.transpose(2, 0, 1).copy())


def inputs(img):
    """The four 1536 px inputs: (part, tile offset mode) -> CHW tensor."""
    bgr = img[..., ::-1]  # part 2 read slides with BGR channel order
    return {
        ("b1", 0): mosaic(get_tiles(img, 192, 64, 0), column_major=True),
        ("b1", 2): mosaic(get_tiles(img, 192, 64, 2), column_major=True),
        ("b0", 0): mosaic(get_tiles(bgr, 256, 36, 0), column_major=False),
        ("b0", 2): mosaic(get_tiles(bgr, 256, 36, 2), column_major=False),
    }


# --------------------------------------------------------------------------- inference

def tta_views(x, tta):
    """The 12 flip/rotation views the ttach wrapper made (or just x)."""
    if not tta:
        return [x]
    views = []
    for h in (False, True):
        for v in (False, True):
            for k in (0, 1, 2):
                y = x.flip(-1) if h else x
                y = y.flip(-2) if v else y
                views.append(torch.rot90(y, k, (-2, -1)))
    return views


@torch.no_grad()
def logits(model, x, tta, batch):
    views = tta_views(x, tta)
    out = [model(torch.stack(views[i:i + batch])) for i in range(0, len(views), batch)]
    return torch.cat(out).mean(0)  # ttach merges logits by mean


def predict_slide(img, b1, b0, use_b1, use_b0, tta, batch, holdout):
    x = inputs(img)
    # Part 1: per fold, sum of 5 ordinal sigmoids, averaged over the two tile modes.
    p1 = [np.mean([logits(b1[k], x["b1", m], tta, batch).sigmoid().sum().item() for m in (0, 2)])
          for k in use_b1]
    part1 = float(np.mean(p1)) if p1 else None
    # Part 2: sigmoids per tile mode for each fold.
    sig = {f: {m: logits(b0[f], x["b0", m], tta, batch).sigmoid() for m in (0, 2)} for f in use_b0}
    if not sig:
        part2 = None
    elif holdout:
        part2 = float(np.mean([(s[0] + s[2]).sum().item() / 2 for s in sig.values()]))
    else:
        # As submitted: fold 0 alone and the 5-fold mean get equal weight.
        mean5 = {m: torch.stack([sig[f][m] for f in B0_FOLDS]).mean(0) for m in (0, 2)}
        part2 = ((sig[0][0] + mean5[0] + sig[0][2] + mean5[2]) / 4).sum().item()
    parts = [p for p in (part1, part2) if p is not None]
    score = float(np.mean(parts))
    return score, part1, part2


# --------------------------------------------------------------------------- held-out folds

def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def training_folds(pnd_dir):
    """image_id -> fold for the b1 and the b0 training sets (absent = never trained on)."""
    rows = read_csv(Path(pnd_dir) / B1_TRAIN_CSV)
    b1 = {r["image_id"]: int(r["kfold"]) for r in rows}
    # The b0 folds were never saved; train_famdata-kfolds.py made them like this.
    from sklearn.model_selection import StratifiedKFold
    rows = read_csv(Path(pnd_dir) / B0_TRAIN_CSV)
    ids = [r["image_id"] for r in rows]
    y = [int(r["isup_grade"]) for r in rows]
    b0 = {}
    for fold, (_, valid) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(ids, y)):
        for i in valid:
            b0[ids[i]] = fold
    return b1, b0


def unseen(image_id, folds_b1, folds_b0):
    use_b1 = [k for k in B1_FOLDS if folds_b1.get(image_id, k) == k]
    use_b0 = [f for f in B0_FOLDS if folds_b0.get(image_id, f) == f]
    return use_b1, use_b0


# --------------------------------------------------------------------------- main

def find_slides(args):
    found = []
    for s in args.slides:
        p = Path(s)
        if p.is_dir():
            found += sorted(p.glob(args.pattern))
        else:
            found += [Path(g) for g in sorted(glob.glob(s))] or [p]
    return [p for p in found if p.is_file()]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("slides", nargs="+", help="slide files, folders or patterns")
    ap.add_argument("--pnd-dir", required=True, help="your clone of Kaggle-PANDA-1st-place-solution")
    ap.add_argument("--out", default=os.path.join("tiles", "pnd_slides.csv"))
    ap.add_argument("--train-csv", default="train.csv", help="PANDA train.csv, adds true grades")
    ap.add_argument("--pattern", default="*.tiff")
    ap.add_argument("--holdout", action="store_true", help="only models that never saw each slide")
    ap.add_argument("--no-tta", action="store_true", help="1 view instead of 12 (12x faster)")
    ap.add_argument("--fast", action="store_true", help="b0 fold 0 + b1 fold 1 only, no TTA")
    ap.add_argument("--batch", type=int, default=4, help="views per forward pass")
    ap.add_argument("--threads", type=int, default=0, help="CPU threads (0 = torch default)")
    ap.add_argument("--redo", action="store_true", help="recompute slides already in --out")
    args = ap.parse_args(argv)

    if args.threads:
        torch.set_num_threads(args.threads)
    slides = find_slides(args)
    if not slides:
        sys.exit("no slides found")
    truth = {r["image_id"]: r for r in read_csv(args.train_csv)} if Path(args.train_csv).exists() else {}
    out = Path(args.out)
    done = {r["image_id"]: r for r in read_csv(out)} if out.exists() and not args.redo else {}
    folds_b1, folds_b0 = training_folds(args.pnd_dir) if args.holdout else ({}, {})
    tta = not (args.no_tta or args.fast)

    print(f"loading models from {args.pnd_dir} ...", flush=True)
    b1, b0 = load_models(args.pnd_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"{len(slides)} slides, TTA {'on' if tta else 'off'}, "
          f"{'held-out models only' if args.holdout else 'one b1 + one b0 model' if args.fast else 'full ensemble'}", flush=True)

    rows = dict(done)
    for n, path in enumerate(slides, 1):
        image_id = path.name.split(".")[0]
        if image_id in done:
            print(f"[{n}/{len(slides)}] {image_id} already done", flush=True)
            continue
        use_b1, use_b0 = (unseen(image_id, folds_b1, folds_b0) if args.holdout
                          else (list(B1_FOLDS), list(B0_FOLDS)))
        if args.fast:
            use_b1, use_b0 = [k for k in use_b1 if k == use_b1[0]], [f for f in use_b0 if f == use_b0[0]]
        t = truth.get(image_id, {})
        if not use_b1 and not use_b0:
            print(f"[{n}/{len(slides)}] {image_id} skipped: every model trained on it", flush=True)
            continue
        start = time.time()
        img = read_level1(path)
        score, p1, p2 = predict_slide(img, b1, b0, use_b1, use_b0, tta, args.batch,
                                      args.holdout or args.fast)
        secs = time.time() - start
        grade = int(np.clip(np.round(score), 0, 5))
        models = " ".join([f"b1f{k}" for k in use_b1] + [f"b0f{f}" for f in use_b0])
        rows[image_id] = dict(zip(HEADER, [
            image_id, grade, f"{score:.3f}", "" if p1 is None else f"{p1:.3f}",
            "" if p2 is None else f"{p2:.3f}", models, t.get("isup_grade", ""),
            t.get("gleason_score", ""), t.get("data_provider", ""), f"{secs:.0f}"]))
        true = f"  (true {t['isup_grade']})" if t else ""
        print(f"[{n}/{len(slides)}] {image_id}  ISUP {grade}  score {score:.2f}{true}  {secs:.0f}s  [{models}]",
              flush=True)
        with open(out, "w", newline="", encoding="utf-8") as f:  # save after every slide
            w = csv.DictWriter(f, HEADER)
            w.writeheader()
            w.writerows(rows.values())

    scored = [r for r in rows.values() if r["isup_grade"] != ""]
    if len(scored) >= 2:
        from sklearn.metrics import cohen_kappa_score
        yt = [int(r["isup_grade"]) for r in scored]
        yp = [int(r["pnd_isup"]) for r in scored]
        print(f"\n{len(scored)} slides with true grades: exact {np.mean(np.equal(yt, yp)):.0%}, "
              f"QWK {cohen_kappa_score(yt, yp, weights='quadratic'):.3f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
