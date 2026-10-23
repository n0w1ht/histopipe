# histopipe — HistoEncoder slide-to-clusters pipeline for Emacs

histopipe runs the whole [HistoEncoder](https://github.com/jopo666/HistoEncoder) workflow from Emacs. It turns whole-slide images into groups of tiles that look alike, and shows you what each group contains:

1. **Download** prostate slides and their label masks from the [PANDA](https://www.kaggle.com/c/prostate-cancer-grade-assessment) Kaggle competition.
2. **Tile**: HistoPrep cuts each slide into small tile images of tissue.
3. **Extract**: HistoEncoder turns every tile into a feature vector.
4. **Cluster**: HistoEncoder groups the tiles of all slides into clusters of similar tissue.
5. **View and name**: for each slide, histopipe draws example tiles from every cluster and colours the slide thumbnail by cluster. You give each cluster a name, such as "stroma" or "tumour glands".

Every step runs in the background, and its output appears in the `*histopipe*` buffer. Each step only does the work that is still missing, so you can add slides or stop halfway and simply run it again.

HistoEncoder is a research tool. Clusters group tiles that *look* alike; they are not diagnoses or Gleason grades.

---

## Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Folders and files](#folders-and-files)
- [Downloading slides from Kaggle](#downloading-slides-from-kaggle)
- [Running the pipeline](#running-the-pipeline)
- [The slide list](#the-slide-list)
- [Cluster views](#cluster-views)
- [Naming clusters](#naming-clusters)
- [Parquet to CSV](#parquet-to-csv)
- [Gleason classifier](#gleason-classifier)
- [Key reference](#key-reference)
- [Customization](#customization)
- [Running without a GPU](#running-without-a-gpu)
- [Troubleshooting](#troubleshooting)
- [Running the tests](#running-the-tests)
- [License](#license)

---

## Requirements

- **Emacs 27.1** or newer.
- **Python 3.11** in a virtual environment (venv). Python 3.12 and newer cannot install HistoEncoder, because it needs torchvision 0.15.
- In that venv:
  - **PyTorch 2.0.1 and torchvision 0.15.2**, installed first.
  - **histoencoder**, which brings HistoPrep, polars and Pillow with it.
  - **polars below 1.0**. HistoEncoder's cluster step uses a polars function that was removed in 1.0.
  - **kaggle**, only if you download slides with histopipe.

Set up the venv on Windows like this (in Command Prompt, from your working folder):

```bat
py -3.11 -m venv histo-env
histo-env\Scripts\activate

rem No NVIDIA GPU:
pip install torch==2.0.1 torchvision==0.15.2
rem NVIDIA GPU instead:
rem pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

pip install histoencoder "polars<1" kaggle
python -c "import histoencoder.functional as F; print('ok')"
```

You don't need to activate the venv to use histopipe. It runs the programs in the venv directly.

## Installation

Save `histopipe.el` in `~/.emacs.d/lisp/` and add this to your init file. Change the paths to your own folders; forward slashes work on Windows.

```elisp
(use-package histopipe
  :load-path "~/.emacs.d/lisp/"
  :commands (histopipe-run histopipe-status
             histopipe-download-slides histopipe-download-sample)
  :bind-keymap ("C-c h" . histopipe-command-map)
  :custom
  (histopipe-venv             "~/prostate/histo-env")
  (histopipe-slides-directory "~/prostate/slides")
  (histopipe-tiles-directory  "~/prostate/tiles")
  (histopipe-masks-directory  "~/prostate/masks")
  (histopipe-train-csv        "~/prostate/train.csv")
  (histopipe-use-cpu t))      ; leave out if you have an NVIDIA GPU
```

Without use-package:

```elisp
(add-to-list 'load-path "~/.emacs.d/lisp/")
(require 'histopipe)
(setq histopipe-venv "~/prostate/histo-env"
      histopipe-slides-directory "~/prostate/slides"
      histopipe-tiles-directory "~/prostate/tiles")
(global-set-key (kbd "C-c h") histopipe-command-map)
```

## Quick start

1. **`C-c h D`**, then 5: download five Radboud slides with their masks.
2. **`C-c h r`**: tile, extract and cluster them. Watch the `*histopipe*` buffer.
3. **`C-c h s`**: open the slide list. Move to a slide and press **`v`** to see its cluster views.
4. Press **`n`** on the same slide to name the clusters as you flip through them.

## Folders and files

histopipe works with these folders. The tiles folder fills up as the pipeline runs.

```
prostate/
├── histo-env/                       the Python venv
├── train.csv                        PANDA slide list (id, provider, ISUP grade, Gleason score)
├── slides/<id>.tiff                 whole-slide images
├── masks/<id>_mask.tiff             label masks
└── tiles/
    ├── cluster_names.csv            your names for the clusters
    └── <id>/                        one folder per slide
        ├── tiles/                   the tile images
        ├── metadata.parquet         HistoPrep: tile positions
        ├── thumbnail.jpeg           HistoPrep: small picture of the slide
        ├── thumbnail_tiles.jpeg     HistoPrep: the same with the tiles drawn on
        ├── features.parquet         HistoEncoder: one feature vector per tile
        ├── clusters.parquet         HistoEncoder: the cluster of each tile
        └── cluster_views/           histopipe: grids, overlay and summary
```

## Downloading slides from Kaggle

PANDA slides are large (often 100 MB or more each), so download a handful at a time.

**One-time setup:**

1. On Kaggle, open *Settings → API → Create New Token*. Save the `kaggle.json` file it gives you in `~/.kaggle/` (on Windows, `C:\Users\<you>\.kaggle\`).
2. Accept the [competition rules](https://www.kaggle.com/c/prostate-cancer-grade-assessment/rules). Without this, Kaggle refuses every download.

**Commands:**

| Key | Command | What it does |
|-----|---------|--------------|
| `C-c h d` | `histopipe-download-slides` | Downloads the slides you name, with their masks. Paste ids or file names, such as `0005f7aaab2800f6170c399693a96917.tiff`, separated by spaces or commas. An id under the cursor is offered as the default. |
| `C-u C-c h d` | | The same without masks (or with them, if `histopipe-download-masks` is nil). |
| `C-c h D` | `histopipe-download-sample` | Downloads `train.csv` if needed, then picks N slides you don't have yet from one data provider, spread evenly over ISUP grades 0 to 5. |

Choose **radboud** for Gleason work: its masks mark stroma, benign tissue and Gleason patterns 3, 4 and 5. **karolinska** masks only mark benign versus cancer.

Files you already have are skipped. A slide without a mask is noted in the log, and the other downloads go on. If Kaggle sends a file zipped, histopipe unpacks it.

## Running the pipeline

| Key | Command | What it runs |
|-----|---------|--------------|
| `C-c h r` | `histopipe-run` | Tile, extract and cluster, then cluster views. |
| `C-u C-c h r` | | Asks for the slides and tiles folders first. |
| `C-u C-u C-c h r` | | Also redoes slides that are already tiled or extracted. |
| `C-c h t` | `histopipe-tile` | Only tiling. |
| `C-c h e` | `histopipe-extract` | Only feature extraction. |
| `C-c h c` | `histopipe-cluster` | Only clustering. With `C-u`, asks for the numbers of clusters. |
| `C-c h k` | `histopipe-stop` | Stops the running step and the rest of the queue. |

What each step skips:

- **Tile** skips slides whose folder already has `metadata.parquet`. A slide whose tiling was interrupted is tiled again.
- **Extract** skips slide folders that already have `features.parquet`.
- **Cluster** always redoes **all** slides together. That keeps cluster numbers consistent: cluster 3 is the same kind of tissue on every slide.

Clustering is done for each number in `histopipe-clusters` (8, 16, 32 and 64 by default). Each becomes a column of `clusters.parquet`, named `n_clusters=8`, `n_clusters=16` and so on.

## The slide list

**`C-c h s`** lists every slide, downloaded or tiled, with how far the pipeline got:

```
Slide                                   Tiles  Features  Clusters
000920ad0b612851f8e01bcc880d9b3d         2981  yes       yes
0005f7aaab2800f6170c399693a96917            -  -         -
```

A `-` under Tiles means the slide is downloaded but not tiled yet.

| Key | What it does on the slide under the cursor |
|-----|--------------------------------------------|
| `RET` | Opens its folder in Dired. |
| `t` | Shows the thumbnail with its tiles drawn on. |
| `v` | Makes its cluster views and opens them. `C-u v` asks which clustering to show. |
| `n` | Names its clusters, one grid at a time. |
| `x` | Writes all its parquet files as CSV. |
| `r` | Runs the pipeline. |
| `k` | Stops the pipeline. |
| `l` | Shows the `*histopipe*` log. |
| `g` | Refreshes the list. |

## Cluster views

After clustering, histopipe writes a `cluster_views` folder into every slide folder (turn this off with `histopipe-views-after-cluster`):

- **`cluster_0.jpg` … `cluster_7.jpg`**: a grid of 16 example tiles from each cluster, with the cluster's name and tile count on top. Flip through these to see what each cluster is.
- **`overlay.jpg`**: the slide thumbnail with every tile coloured by its cluster, and a legend, so you can see where each pattern sits on the tissue.
- **`summary.csv`**: one row per cluster with its name, number of tiles and percentage of the slide, largest first.

```csv
cluster,name,tiles,percent
0,stroma,1204,40.4
3,"tumour glands",611,20.5
5,,402,13.5
```

The views show `histopipe-view-column`, which is `n_clusters=8` by default. To look at a finer split, press `C-u v` and pick `n_clusters=16`. `C-c h V` remakes the views of every slide, for example after renaming clusters.

## Naming clusters

Press **`n`** on a slide in the slide list. histopipe shows each cluster's grid next to the minibuffer and asks for a name, such as "benign glands", "stroma", "tumour glands" or "edge/fold artifact".

- **RET** keeps the name the cluster already has.
- **`-`** removes the name.

The names are saved in `tiles/cluster_names.csv` and apply to **every** slide, since cluster numbers are shared. After naming, the slide's views are redone so the legend and summary show the names. Press `C-c h V` to update the other slides.

Names belong to one clustering: names for `n_clusters=8` don't carry over to `n_clusters=16`. **If you add slides and cluster again, the cluster numbers change**, so name the clusters again.

## Parquet to CSV

Parquet files can't be opened in a text editor or Excel. histopipe writes them as CSV next to the original: `features.parquet` becomes `features.csv`.

- **`C-c h x`** converts a file you choose. In Dired, it converts the marked parquet files.
- **`x`** in the slide list converts all parquet files of that slide.

To make the CSV easier to read, the full tile path becomes just the tile's file name, in the first column, and decimals are rounded to 4 places (`histopipe-csv-decimals`). A single converted file opens in Emacs when it's ready.

`features.csv` is large: one row per tile and 384 feature columns.

## Gleason classifier

`gleason/gleason.py` trains a small classifier on top of HistoEncoder features to predict a Gleason pattern for every tile, and a Gleason score and ISUP grade for every slide. It uses histopipe's output folders and the Radboud label masks, and runs on a CPU in minutes. It also needs `pip install openslide-python openslide-bin` in the venv, to read the masks.

From your working folder, with the venv active, after `C-c h D` (radboud slides with masks) and `C-c h r`:

```bat
python gleason\gleason.py label      & rem tile labels from the masks -> tiles\<id>\labels.csv
python gleason\gleason.py train      & rem logistic regression; tests on 20% held-out slides
python gleason\gleason.py predict    & rem gleason_tiles.csv + gleason_map.jpg per slide, tiles\gleason_slides.csv
python gleason\gleason.py evaluate   & rem predicted vs. true ISUP grade from train.csv
```

The slide score follows the biopsy rule: the most common cancer pattern plus the highest other pattern. `python gleason\gleason.py <command> --help` lists the options. This is a research model, not a diagnostic tool.

## Key reference

Bind `histopipe-command-map` to a prefix, here `C-c h`:

| Key | Command |
|-----|---------|
| `C-c h r` | `histopipe-run`: the whole pipeline |
| `C-c h t` / `e` / `c` | tile / extract / cluster only |
| `C-c h k` | `histopipe-stop` |
| `C-c h s` | `histopipe-status`: the slide list |
| `C-c h v` | `histopipe-view-slide`: cluster views of a slide |
| `C-c h V` | `histopipe-view-all`: cluster views of every slide |
| `C-c h n` | `histopipe-name-clusters` |
| `C-c h d` | `histopipe-download-slides` |
| `C-c h D` | `histopipe-download-sample` |
| `C-c h x` | `histopipe-parquet-to-csv` |
| `C-c h p` | `histopipe-pnd-predict-all`: grade every slide with the PANDA ensemble |
| `C-c h P` | `histopipe-pnd-results`: open `pnd_slides.csv` |

In the `*histopipe*` log, `k` stops the pipeline and `s` opens the slide list.

## Customization

`M-x customize-group RET histopipe` shows all options.

| Option | Default | Meaning |
|--------|---------|---------|
| **Folders** | | |
| `histopipe-venv` | `"~/histo/histo-env"` | Python venv with histoencoder |
| `histopipe-slides-directory` | `"~/histo/slides"` | Whole-slide images |
| `histopipe-slide-pattern` | `"*.tiff"` | Which files in it are slides |
| `histopipe-tiles-directory` | `"~/histo/tiles"` | Pipeline output, one folder per slide |
| `histopipe-masks-directory` | `"~/histo/masks"` | Downloaded label masks |
| `histopipe-train-csv` | `"~/histo/train.csv"` | PANDA slide list |
| **Tiling** | | |
| `histopipe-level` | `0` | Pyramid level to cut tiles from (0 = full resolution) |
| `histopipe-tile-width` | `512` | Tile size in pixels |
| `histopipe-tile-overlap` | `0.5` | Overlap between neighbouring tiles, 0 to 1 |
| `histopipe-max-background` | `0.5` | Largest share of background a kept tile may have |
| **Features** | | |
| `histopipe-model` | `"prostate_small"` | `prostate_small` or `prostate_medium` |
| `histopipe-num-blocks` | `1` | Attention blocks whose outputs make up the features |
| `histopipe-avg-pool` | `nil` | Also append average-pooled features |
| `histopipe-batch-size` | `16` | Tiles per batch |
| `histopipe-workers` | `4` | Data loading processes. All cores can hang on Windows. |
| `histopipe-use-cpu` | `nil` | `t` to skip the GPU check |
| **Clusters and views** | | |
| `histopipe-clusters` | `(8 16 32 64)` | Numbers of clusters |
| `histopipe-view-column` | `"n_clusters=8"` | Clustering shown in the views |
| `histopipe-view-tiles-per-cluster` | `16` | Example tiles per grid |
| `histopipe-view-tile-px` | `128` | Size of each example tile |
| `histopipe-views-after-cluster` | `t` | Make views automatically after clustering |
| `histopipe-view-script` | `nil` | Your own view script instead of the built-in views |
| **Downloads and CSV** | | |
| `histopipe-kaggle-competition` | `"prostate-cancer-grade-assessment"` | Kaggle competition |
| `histopipe-download-masks` | `t` | Download masks with slides |
| `histopipe-csv-decimals` | `4` | Decimals kept in CSV files |

## Running without a GPU

Everything works on the CPU; feature extraction is the slow part. To keep it manageable:

- Set `histopipe-use-cpu` to `t`.
- Set `histopipe-tile-overlap` to `0.0`. That makes about 4 times fewer tiles than 0.5, and still covers all the tissue.
- Keep `prostate_small`. `prostate_medium` is bigger and slower.
- Time one slide first: the log shows "extract done in" for each run.

## Troubleshooting

**`pip install histoencoder` says "Cannot install … conflicting dependencies … torchvision".**
Your Python is 3.12 or newer. Make the venv with Python 3.11, as in [Requirements](#requirements).

**The cluster step fails with "has no attribute 'groupby'".**
Your polars is 1.0 or newer. Run `pip install "polars<1"` in the venv. The log suggests this too.

**Tiling fails with "Got unexpected extra arguments".**
You have an older histopipe.el that ran `HistoPrep.exe` directly. On Windows that program expands `*.tiff` into a list of file names before reading it. Update histopipe.el; it now starts HistoPrep through Python.

**A download fails with 401 or 403.**
`kaggle.json` is missing from `~/.kaggle/`, or you haven't accepted the competition rules. See [Downloading slides from Kaggle](#downloading-slides-from-kaggle).

**"… is not tiled yet" when pressing `x`, `v` or `RET` in the slide list.**
The slide is downloaded but hasn't been through the pipeline. Run `C-c h r`.

**Extraction seems stuck on Windows.**
Lower `histopipe-workers` to 2, or to 0 to load data in the main process.

**`overlay.jpg` looks shifted or the wrong size.**
histopipe reads the slide size through HistoPrep. If the slide file can't be found or opened, it estimates the size and says so in the log. Keep the slide in the slides folder under its original name.

## Running the tests

The tests live in `test/`, with stand-in versions of HistoPrep, HistoEncoder and kaggle in `test/w/venv/`, so they need no real slides, model or downloads. The stand-ins are shell scripts, so the tests run on Linux or macOS (or WSL), not in Windows Emacs. From the repository root:

```sh
emacs -Q --batch -L . -l test/histopipe-test.el -f ert-run-tests-batch-and-exit
```

Two tests use real polars and Pillow and are skipped unless you point them at a venv that has both, and at test data:

- `HISTOPIPE_POLARS_VENVS`: one or more venvs, separated by `:`
- `HISTOPIPE_TEST_PARQUET`: a folder with `features.parquet` and `clusters.parquet`
- `HISTOPIPE_TEST_VIEWS`: a slide folder with `tiles/`, `clusters.parquet` and `thumbnail.jpeg`

## PANDA 1st-place ensemble

`pnd/pnd_predict.py` runs the model ensemble that won the PANDA challenge (team PND) on your slides, on a CPU, as a benchmark for the Gleason classifier. It uses their published weights, which are licensed CC-BY-NC, for non-commercial use only. See [pnd/README.md](pnd/README.md) for setup and options.

From Emacs:

| Key | What it does |
| --- | --- |
| `C-c h p` | Grades every slide. Slides that are already graded are skipped, so a stopped run continues where it left off. |
| `C-u C-c h p` | Asks for the speed, whether to use only held-out models, and whether to regrade, then grades every slide. |
| `p` in the slide list | Grades the slide on that line (it is regraded if it was graded before). |
| `C-c h P` | Opens the results, `tiles/pnd_slides.csv`. |

The slide list gains a **PND** column with the predicted ISUP grade and a **True** column with the grade from train.csv. The first run clones their repository, which holds the weights (about 350 MB), into the folder above the tiles folder. It needs `git`; without git, download the repository's ZIP and unzip it there as `Kaggle-PANDA-1st-place-solution`. The venv needs the packages from [pnd/README.md](pnd/README.md#setup-windows-cpu).

| Option | Default | Meaning |
| --- | --- | --- |
| `histopipe-pnd-speed` | `full` | `full` (12-view TTA, minutes per slide), `no-tta` (under a minute) or `fast` (one model of each kind, seconds) |
| `histopipe-pnd-holdout` | `t` | use only the models that never trained on each slide; keep it on for PANDA train slides |
| `histopipe-pnd-directory` | `nil` | the repository with the weights; `nil` means `Kaggle-PANDA-1st-place-solution` next to the tiles folder |
| `histopipe-pnd-script` | `nil` | `pnd_predict.py`; `nil` finds `pnd/pnd_predict.py` next to histopipe.el, or `pnd_predict.py` next to the tiles folder |
| `histopipe-pnd-threads` | `nil` | CPU threads for torch |

From the command line:

```bat
python pnd\pnd_predict.py slides --pnd-dir Kaggle-PANDA-1st-place-solution --holdout
```

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).
