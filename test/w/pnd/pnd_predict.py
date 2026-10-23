# Stub of pnd/pnd_predict.py: echoes its arguments and writes one row per slide.
import argparse, glob, os, sys
print("args:", " ".join(sys.argv[1:]))
ap = argparse.ArgumentParser()
ap.add_argument("slides", nargs="+"); ap.add_argument("--pnd-dir"); ap.add_argument("--out")
ap.add_argument("--train-csv"); ap.add_argument("--pattern"); ap.add_argument("--threads")
for f in ("--holdout", "--no-tta", "--fast", "--redo"): ap.add_argument(f, action="store_true")
a = ap.parse_args()
assert os.path.isdir(os.path.join(a.pnd_dir, "final_models"))
files = []
for s in a.slides:
    files += sorted(glob.glob(os.path.join(s, a.pattern))) if os.path.isdir(s) else [s]
with open(a.out, "w") as f:
    f.write("image_id,pnd_isup,pnd_score,part_b1,part_b0,models,isup_grade,gleason_score,data_provider,seconds\n")
    for i, p in enumerate(files):
        f.write(f"{os.path.basename(p).split('.')[0]},{i + 2},{i + 2}.1,,,b0f0,2,3+3,radboud,1\n")
