r"""
gleason.py - a small Gleason classifier on top of HistoEncoder features.

Run from your prostate folder with histo-env active, after histopipe (or
HistoPrep + HistoEncoder extract) has made tiles\<id>\features.parquet:

    python gleason.py label     tile labels from the PANDA Radboud masks
    python gleason.py train     fit the classifier, test it on held-out slides
    python gleason.py predict   Gleason pattern per tile and score per slide
    python gleason.py evaluate  compare predicted ISUP grades with train.csv

Folders default to .\tiles, .\masks, .\slides and .\train.csv; see
`python gleason.py <command> --help`.

Mask values (Radboud, red channel): 0 background, 1 stroma, 2 benign
epithelium, 3/4/5 Gleason pattern 3/4/5.

Research use only: this is not a validated diagnostic tool.
"""
import argparse
import csv
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import polars as pl

CLASSES = {1: "stroma", 2: "benign", 3: "G3", 4: "G4", 5: "G5"}
COLORS = {1: (150, 150, 150), 2: (60, 180, 75), 3: (255, 225, 25), 4: (245, 130, 48), 5: (230, 25, 75)}
ISUP = {(3, 3): 1, (3, 4): 2, (4, 3): 3, (4, 4): 4, (3, 5): 4, (5, 3): 4,
        (4, 5): 5, (5, 4): 5, (5, 5): 5}


# --------------------------------------------------------------------------- helpers

def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def slide_dirs(tiles, needed="features.parquet"):
    return sorted(d for d in Path(tiles).iterdir() if (d / needed).exists())


def providers(train_csv):
    if not Path(train_csv).exists():
        return {}
    return {r["image_id"]: r for r in read_csv(train_csv)}


def load_features(slide_dir):
    """(x, y) of each tile and its feature matrix."""
    df = pl.read_parquet(slide_dir / "features.parquet")
    feats = [c for c in df.columns if c.startswith("feat")]
    xy = list(zip(df["x"].to_list(), df["y"].to_list()))
    return xy, df.select(feats).to_numpy().astype(np.float32), df


def read_mask(path, max_dim=8192):
    """Red channel of a label mask at a pyramid level no larger than max_dim,
    and its downsample factor from level 0."""
    try:
        import openslide
        s = openslide.OpenSlide(str(path))
        level = next((i for i, (w, h) in enumerate(s.level_dimensions) if max(w, h) <= max_dim),
                     s.level_count - 1)
        w, h = s.level_dimensions[level]
        region = s.read_region((0, 0), level, (w, h))
        return np.asarray(region)[:, :, 0], s.level_downsamples[level]
    except ImportError:
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        im = Image.open(path)
        full_w = im.width
        factor = max(1, math.ceil(max(im.size) / max_dim))
        if factor > 1:
            im = im.resize((im.width // factor, im.height // factor), Image.NEAREST)
        return np.asarray(im.convert("RGB"))[:, :, 0], full_w / im.width


def slide_width(slides, slide_id):
    for suffix in (".tiff", ".tif", ".svs", ".ndpi", ".mrxs"):
        p = Path(slides) / f"{slide_id}{suffix}"
        if p.exists():
            try:
                from histoprep import SlideReader
                return SlideReader(str(p)).dimensions[1]
            except Exception:
                pass
    return None


# --------------------------------------------------------------------------- label

def cmd_label(a):
    info = providers(a.train_csv)
    done = 0
    for d in slide_dirs(a.tiles):
        sid = d.name
        mask_file = Path(a.masks) / f"{sid}_mask.tiff"
        if info and info.get(sid, {}).get("data_provider") != "radboud":
            print(f"{sid}: not a Radboud slide, skipped (its mask has no Gleason patterns)")
            continue
        if not mask_file.exists():
            print(f"{sid}: no mask {mask_file}, skipped")
            continue
        if (d / "labels.csv").exists() and not a.overwrite:
            print(f"{sid}: labels.csv exists, skipped (use --overwrite)")
            continue
        mask, ds = read_mask(mask_file)
        _, _, df = load_features(d)
        rows = []
        for x, y, w, h in zip(*(df[c].to_list() for c in "xywh")):
            x0, y0 = int(x / ds), int(y / ds)
            x1, y1 = max(x0 + 1, int((x + w) / ds)), max(y0 + 1, int((y + h) / ds))
            patch = mask[y0:y1, x0:x1]
            counts = np.bincount(patch.ravel(), minlength=6)[:6] if patch.size else np.zeros(6, int)
            tissue = counts[1:].sum()
            if tissue == 0:
                continue
            label = int(np.argmax(counts[1:]) + 1)
            rows.append([x, y, w, h, label, CLASSES[label],
                         round(tissue / max(1, patch.size), 3), round(counts[label] / tissue, 3)])
        write_csv(d / "labels.csv", ["x", "y", "w", "h", "label", "class", "tissue", "purity"], rows)
        summary = {CLASSES[k]: sum(r[4] == k for r in rows) for k in CLASSES}
        print(f"{sid}: {len(rows)} labelled tiles  {summary}")
        done += 1
    print(f"\nLabelled {done} slides.")


# --------------------------------------------------------------------------- train

def labelled_tiles(d, min_tissue, min_purity):
    xy, X, _ = load_features(d)
    index = {p: i for i, p in enumerate(xy)}
    keep, y = [], []
    for r in read_csv(d / "labels.csv"):
        i = index.get((int(r["x"]), int(r["y"])))
        if i is None or float(r["tissue"]) < min_tissue or float(r["purity"]) < min_purity:
            continue
        keep.append(i)
        y.append(int(r["label"]))
    return X[keep], np.array(y, dtype=int)


def cmd_train(a):
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    dirs = slide_dirs(a.tiles, "labels.csv")
    if len(dirs) < 2:
        sys.exit("Need labels.csv for at least 2 slides; run `python gleason.py label` first.")
    random.seed(a.seed)
    ids = [d.name for d in dirs]
    random.shuffle(ids)
    n_test = 0 if a.all else max(1, round(len(ids) * a.test_fraction))
    test_ids, train_ids = set(ids[:n_test]), set(ids[n_test:])

    def stack(which):
        parts = [labelled_tiles(d, a.min_tissue, a.min_purity) for d in dirs if d.name in which]
        parts = [p for p in parts if len(p[1])]
        if not parts:
            return np.zeros((0, 1)), np.zeros(0, int)
        return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])

    X, y = stack(train_ids)
    print(f"Training on {len(train_ids)} slides, {len(y)} tiles: "
          f"{ {CLASSES[k]: int((y == k).sum()) for k in CLASSES} }")
    model = make_pipeline(StandardScaler(),
                          LogisticRegression(C=a.C, class_weight="balanced", max_iter=3000))
    model.fit(X, y)

    if test_ids:
        Xt, yt = stack(test_ids)
        if len(yt):
            pred = model.predict(Xt)
            labels = [k for k in CLASSES if k in set(yt) | set(pred)]
            print(f"\nTest: {len(test_ids)} held-out slides, {len(yt)} tiles")
            print(classification_report(yt, pred, labels=labels,
                                        target_names=[CLASSES[k] for k in labels], zero_division=0))
            print("Confusion matrix (rows = mask label, columns = prediction):")
            print("        " + " ".join(f"{CLASSES[k]:>7}" for k in labels))
            for k, row in zip(labels, confusion_matrix(yt, pred, labels=labels)):
                print(f"{CLASSES[k]:>7} " + " ".join(f"{v:>7}" for v in row))
    joblib.dump({"model": model, "train_ids": sorted(train_ids), "test_ids": sorted(test_ids),
                 "classes": CLASSES}, a.model)
    print(f"\nSaved {a.model}  (held-out slides: {', '.join(sorted(test_ids)) or 'none'})")


# --------------------------------------------------------------------------- predict

def gleason_from_counts(counts, min_tumour_tiles, min_share):
    """Biopsy-style score: most common pattern + highest other pattern.
    Patterns below min_share of tumour tiles are treated as noise."""
    tumour = {k: counts.get(k, 0) for k in (3, 4, 5)}
    total = sum(tumour.values())
    if total < min_tumour_tiles:
        return "negative", 0, total
    present = [k for k, n in tumour.items() if n / total >= min_share] or [max(tumour, key=tumour.get)]
    primary = max(present, key=lambda k: (tumour[k], k))
    others = [k for k in present if k != primary]
    secondary = max(others) if others else primary
    return f"{primary}+{secondary}", ISUP[(primary, secondary)], total


def draw_map(d, rows, slides):
    from PIL import Image, ImageDraw
    thumb_file = d / "thumbnail.jpeg"
    if not thumb_file.exists():
        return
    thumb = Image.open(thumb_file).convert("RGBA")
    width = slide_width(slides, d.name)
    if not width:
        ratio = max(r[0] + r[2] for r in rows) / thumb.width
        width = thumb.width * 2 ** math.ceil(math.log2(max(ratio, 1)))
    s = thumb.width / width
    layer = Image.new("RGBA", thumb.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    for x, y, w, h, k, *_ in rows:
        draw.rectangle([x * s, y * s, (x + w) * s, (y + h) * s], fill=COLORS[k] + (110,))
    out = Image.alpha_composite(thumb, layer).convert("RGB")
    canvas = Image.new("RGB", (out.width + 100, max(out.height, 110)), "white")
    canvas.paste(out, (0, 0))
    dr = ImageDraw.Draw(canvas)
    for i, k in enumerate(CLASSES):
        dr.rectangle([out.width + 10, 10 + 20 * i, out.width + 24, 24 + 20 * i], fill=COLORS[k])
        dr.text((out.width + 30, 10 + 20 * i), CLASSES[k], fill="black")
    canvas.save(d / "gleason_map.jpg", quality=90)


def cmd_predict(a):
    import joblib
    bundle = joblib.load(a.model)
    model = bundle["model"]
    dirs = slide_dirs(a.tiles)
    if a.slides_ids:
        dirs = [d for d in dirs if d.name in set(a.slides_ids)]
    results = []
    for d in dirs:
        xy, X, df = load_features(d)
        if not len(xy):
            continue
        prob = model.predict_proba(X)
        classes = list(model.classes_)
        pred = [classes[i] for i in prob.argmax(axis=1)]
        rows = [[x, y, w, h, k, CLASSES[k]] + [round(float(p), 3) for p in pr]
                for (x, y, w, h), k, pr in zip(zip(*(df[c].to_list() for c in "xywh")), pred, prob)]
        write_csv(d / "gleason_tiles.csv",
                  ["x", "y", "w", "h", "label", "class"] + [f"p_{CLASSES[k]}" for k in classes], rows)
        counts = {k: pred.count(k) for k in CLASSES}
        score, isup, tumour = gleason_from_counts(counts, a.min_tumour_tiles, a.min_share)
        draw_map(d, rows, a.slides)
        status = "training" if d.name in bundle["train_ids"] else "held-out" if d.name in bundle["test_ids"] else "new"
        results.append([d.name, score, isup, tumour, len(pred)]
                       + [counts[k] for k in CLASSES] + [status])
        print(f"{d.name}: Gleason {score:<8} ISUP {isup}  ({tumour} tumour of {len(pred)} tiles, {status} slide)")
    out = Path(a.tiles) / "gleason_slides.csv"
    # Keep earlier results for slides not predicted this time.
    done = {r[0] for r in results}
    if out.exists():
        old = [list(r.values()) for r in read_csv(out) if r["slide"] not in done]
        results = sorted(old + results, key=lambda r: r[0])
    write_csv(out, ["slide", "gleason", "isup", "tumour_tiles", "tiles"]
              + [f"n_{CLASSES[k]}" for k in CLASSES] + ["slide_set"], results)
    print(f"\nWrote {out} and gleason_tiles.csv + gleason_map.jpg in each slide folder.")


# --------------------------------------------------------------------------- evaluate

def cmd_evaluate(a):
    from sklearn.metrics import cohen_kappa_score, confusion_matrix
    truth = providers(a.train_csv)
    if not truth:
        sys.exit(f"{a.train_csv} not found.")
    rows = read_csv(Path(a.tiles) / "gleason_slides.csv")
    sets = ["held-out", "new"] + (["training"] if a.include_training else [])
    rows = [r for r in rows if r["slide_set"] in sets and r["slide"] in truth]
    if not rows:
        sys.exit("No held-out or new slides with a known grade; predict on slides the model "
                 "was not trained on, or use --include-training.")
    yt = [int(truth[r["slide"]]["isup_grade"]) for r in rows]
    yp = [int(r["isup"]) for r in rows]
    print(f"{'slide':<34} {'true':>12} {'predicted':>12}")
    for r, t in zip(rows, yt):
        print(f"{r['slide']:<34} {truth[r['slide']]['gleason_score'] + ' / ' + str(t):>12} "
              f"{r['gleason'] + ' / ' + r['isup']:>12}")
    print(f"\n{len(rows)} slides. Exact ISUP grade: {np.mean(np.array(yt) == np.array(yp)):.0%}. "
          f"Within one grade: {np.mean(np.abs(np.array(yt) - np.array(yp)) <= 1):.0%}.")
    if len(set(yt + yp)) > 1:
        print(f"Quadratic weighted kappa: {cohen_kappa_score(yt, yp, weights='quadratic'):.2f} "
              "(PANDA winners reached about 0.9 with thousands of slides)")
    grades = list(range(6))
    print("\nISUP confusion matrix (rows = true 0..5, columns = predicted 0..5):")
    for g, row in zip(grades, confusion_matrix(yt, yp, labels=grades)):
        print(f"  {g}: " + " ".join(f"{v:>3}" for v in row))


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    def common(p, *, masks=False, model=False, slides=False, train_csv=False):
        p.add_argument("--tiles", default="tiles", help="histopipe tiles folder (default: tiles)")
        if masks:
            p.add_argument("--masks", default="masks", help="folder with <id>_mask.tiff (default: masks)")
        if model:
            p.add_argument("--model", default="gleason_model.joblib", help="model file (default: gleason_model.joblib)")
        if slides:
            p.add_argument("--slides", default="slides", help="slide images, for the map (default: slides)")
        if train_csv:
            p.add_argument("--train-csv", default="train.csv", help="PANDA train.csv (default: train.csv)")

    p = sub.add_parser("label", help="tile labels from Radboud masks")
    common(p, masks=True, train_csv=True)
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_label)

    p = sub.add_parser("train", help="fit the classifier")
    common(p, model=True)
    p.add_argument("--test-fraction", type=float, default=0.2, help="share of slides held out (default 0.2)")
    p.add_argument("--all", action="store_true", help="train on every slide, hold none out")
    p.add_argument("--min-tissue", type=float, default=0.5, help="min tissue share of a tile (default 0.5)")
    p.add_argument("--min-purity", type=float, default=0.6, help="min share of the majority label (default 0.6)")
    p.add_argument("--C", type=float, default=0.1, help="inverse regularisation strength (default 0.1)")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("predict", help="score slides")
    common(p, model=True, slides=True)
    p.add_argument("slides_ids", nargs="*", metavar="SLIDE", help="slide ids (default: all)")
    p.add_argument("--min-tumour-tiles", type=int, default=10, help="fewer tumour tiles = negative (default 10)")
    p.add_argument("--min-share", type=float, default=0.05, help="ignore patterns below this share of tumour (default 0.05)")
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("evaluate", help="compare with train.csv")
    common(p, train_csv=True)
    p.add_argument("--include-training", action="store_true", help="also count slides the model trained on")
    p.set_defaults(func=cmd_evaluate)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
