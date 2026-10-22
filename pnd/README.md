# PANDA 1st-place ensemble on CPU

`pnd_predict.py` runs the model ensemble that won the PANDA challenge (team PND, private QWK 0.940) on your own slides, on a CPU, and writes an ISUP grade for each slide. It reproduces their final Kaggle submission (`submitted_notebook.ipynb`):

| Part | Models | Input |
| --- | --- | --- |
| 1 | EfficientNet-b1, folds 1, 4, 5 | 64 tiles of 192 px from level 1, 8×8 mosaic |
| 2 | EfficientNet-b0, folds 0–4 | 36 tiles of 256 px from level 1, 6×6 mosaic |

Every model sees 12 flipped and rotated copies of each mosaic (test-time augmentation). Each slide is tiled twice, with the tile grid at two offsets. The final score is 0.5 × part 1 + 0.5 × part 2, rounded to an ISUP grade from 0 to 5.

The models and weights are by Kentaro Yoshioka, fam_taro, Arutema and team PND: <https://github.com/kentaroy47/Kaggle-PANDA-1st-place-solution>. They are licensed **CC-BY-NC 4.0**, which allows only non-commercial use and requires credit to the authors. They are research models, not a diagnostic tool.

## Setup (Windows, CPU)

1. Open a Command Prompt in your working folder, for example `C:\Users\Hormoz\prostate`, and get their repository, which includes the weights (about 350 MB):

   ```bat
   git clone https://github.com/kentaroy47/Kaggle-PANDA-1st-place-solution
   ```

   If you don't have git, use **Code → Download ZIP** on GitHub and unzip it into the same folder.

2. Install three packages in the HistoEncoder venv. The scripts already need torch, numpy and scikit-learn, and those are installed. Pin the versions: the newest tifffile and imagecodecs pull in numpy 2, which torch 2.0.1 can't use.

   ```bat
   histo-env\Scripts\activate
   pip install efficientnet_pytorch "tifffile==2024.8.30" "imagecodecs==2024.6.1" "numpy==1.26.4"
   ```

3. Copy `pnd_predict.py` into the folder, or run it from your histopipe checkout.

## Running

```bat
rem every slide in slides\
python pnd_predict.py slides --pnd-dir Kaggle-PANDA-1st-place-solution

rem one slide
python pnd_predict.py slides\0005f7aaab2800f6170c399693a96917.tiff --pnd-dir Kaggle-PANDA-1st-place-solution

rem PANDA train slides, scored fairly (see below)
python pnd_predict.py slides --pnd-dir Kaggle-PANDA-1st-place-solution --holdout
```

Results go to `tiles\pnd_slides.csv`, which is saved after every slide. Slides that are already in the file are skipped, so you can stop a long run with Ctrl+C and start it again later. If `train.csv` is in the folder, the true grades are added and the run ends with accuracy and QWK.

| Column | Meaning |
| --- | --- |
| `pnd_isup` | predicted ISUP grade, 0–5 |
| `pnd_score` | the unrounded score; 2.4 means between grade 2 and 3 |
| `part_b1`, `part_b0` | each half of the ensemble on its own |
| `models` | the models used, such as `b1f4 b0f4` |
| `isup_grade`, `gleason_score`, `data_provider` | from `train.csv`, when the slide is in it |

## Speed

The full ensemble makes 192 forward passes per slide, so expect several minutes per slide. It took 9 minutes on 4 cloud cores. Two options make it faster:

| Option | Passes per slide | Approximate time |
| --- | --- | --- |
| (default) | 192 | 5–10 min |
| `--no-tta` | 16 | under 1 min |
| `--fast` (one b1 and one b0 model, no TTA) | 4 | about 10 s |

`--threads N` sets how many CPU threads torch uses.

## Scoring PANDA train slides fairly: `--holdout`

The models were trained on most of PANDA train, so on those slides the full ensemble is partly marking its own homework, and its accuracy will look too good. `--holdout` uses only the models that never saw each slide:

- **b1:** the fold model whose validation fold contains the slide, from `input/train-5kfold_remove_noisy_…csv`.
- **b0:** the same rule. Their b0 folds were never saved, so the script rebuilds them the way `train_famdata-kfolds.py` made them, with `StratifiedKFold(5, shuffle=True, random_state=42)` on their cleaned label file. This should match, but it can't be checked.
- **Slides removed as noisy:** no model trained on these, so all of them are used. Remember that these labels are the ones the team judged unreliable.

With `--holdout`, a slide usually gets one b1 model or none, plus one b0 model. The `models` column shows which were used. To compare with `gleason.py evaluate`, use `--holdout` on the same slides.

## How faithful is it?

I checked the b1 model against the model code in their notebook, using the real weights on the same input; the outputs match to within 3e-7. The tiling, the mosaic layout (part 1 fills columns first, while part 2 fills rows first and uses BGR channel order), the TTA set and the blend all follow the notebook. The notebook also gives b0 fold 0 double weight, and the script does the same. The notebook only used TTA for part 2 when it had more than 50 slides; this script always uses it unless you pass `--no-tta`.
