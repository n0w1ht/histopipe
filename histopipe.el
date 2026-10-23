;;; histopipe.el --- Run the HistoPrep/HistoEncoder slide-to-clusters pipeline -*- lexical-binding: t; -*-

;; Copyright (C) 2026 Hormoz Zamanpour Siahkal

;; Author: Hormoz Zamanpour Siahkal <n0w1ht@users.noreply.github.com>
;; Version: 0.1.0
;; Package-Requires: ((emacs "27.1"))
;; Keywords: tools, processes

;; This file is not part of GNU Emacs.

;; This program is free software; you can redistribute it and/or modify
;; it under the terms of the GNU General Public License as published by
;; the Free Software Foundation, either version 3 of the License, or
;; (at your option) any later version.

;; This program is distributed in the hope that it will be useful,
;; but WITHOUT ANY WARRANTY; without even the implied warranty of
;; MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
;; GNU General Public License for more details.

;; You should have received a copy of the GNU General Public License
;; along with this program.  If not, see <https://www.gnu.org/licenses/>.

;;; Commentary:

;; histopipe runs the three steps that take whole-slide images to
;; HistoEncoder tile clusters, asynchronously, with the output in the
;; *histopipe* buffer:
;;
;;   1. Tile     HistoPrep cuts slides into tiles
;;   2. Extract  HistoEncoder extracts a feature vector per tile
;;   3. Cluster  HistoEncoder clusters the features of all slides together
;;
;; The executables are taken from a Python virtual environment
;; (`histopipe-venv'), so it does not need to be activated.  Each step
;; only does what is missing: slides that are already tiled or already
;; have features are skipped, so the pipeline can be re-run after adding
;; slides or after an interruption.  Clustering always redoes every
;; slide, so cluster ids mean the same tissue pattern on every slide.
;;
;; Quick start:
;;
;;   (require 'histopipe)
;;   (setq histopipe-venv "C:/histo/histo-env"
;;         histopipe-slides-directory "C:/histo/slides"
;;         histopipe-tiles-directory "C:/histo/tiles")
;;   (global-set-key (kbd "C-c h") histopipe-command-map)
;;
;; Then `C-c h r' runs the whole pipeline (with a prefix argument it
;; asks for the folders first), `C-c h s' shows which slides are done,
;; and `C-c h k' stops a running step.  In the status buffer, RET opens
;; a slide's folder, `t' its tile thumbnail and `x' writes its parquet
;; files as CSV (`C-c h x' converts any parquet file, or the marked
;; ones in Dired).
;;
;; After clustering, each slide folder gets cluster_views/ with
;; cluster_<k>.jpg (a grid of example tiles per cluster), overlay.jpg
;; (the thumbnail coloured by cluster) and summary.csv (tiles and
;; share of the slide per cluster).  `v' on a slide remakes and opens
;; them, `C-c h V' remakes them for every slide, and `n' shows each
;; cluster's grid in turn and asks for a name such as "stroma" or
;; "tumour glands"; names go into the legend and summary.csv.
;;
;; Slides can be downloaded from the PANDA Kaggle competition with the
;; kaggle CLI (pip install kaggle in the same venv, kaggle.json in
;; ~/.kaggle/, competition rules accepted on the website):
;; `C-c h d' downloads slides by id, such as
;; 0005f7aaab2800f6170c399693a96917, with their label masks, and
;; `C-c h D' picks N slides of one data provider (radboud by default)
;; spread over ISUP grades, using train.csv.
;;
;; `C-c h p' grades every slide with the PANDA challenge's winning
;; ensemble (team PND, pnd/pnd_predict.py), cloning their weights
;; first if needed; `p' in the status list grades one slide, and the
;; PND and True columns show predicted and true ISUP grades.
;; `C-c h P' opens the results, tiles/pnd_slides.csv.

;;; Code:

(require 'ansi-color)
(require 'cl-lib)
(require 'subr-x)
(require 'tabulated-list)

(declare-function dired-get-marked-files "dired")

(defgroup histopipe nil
  "Run the HistoPrep/HistoEncoder slide-to-clusters pipeline."
  :group 'tools
  :prefix "histopipe-")

(defcustom histopipe-venv "~/histo/histo-env"
  "Python virtual environment where histoencoder is installed.
When nil, or when an executable is missing there, it is looked up
on `exec-path'."
  :type '(choice (const :tag "Use exec-path" nil) directory))

(defcustom histopipe-slides-directory "~/histo/slides"
  "Folder with the whole-slide images."
  :type 'directory)

(defcustom histopipe-slide-pattern "*.tiff"
  "Wildcard for slide files inside `histopipe-slides-directory'."
  :type 'string)

(defcustom histopipe-tiles-directory "~/histo/tiles"
  "Output folder: HistoPrep makes one subfolder per slide."
  :type 'directory)

(defcustom histopipe-level 0
  "Slide pyramid level to cut tiles from."
  :type 'integer)

(defcustom histopipe-tile-width 512
  "Tile width in pixels."
  :type 'integer)

(defcustom histopipe-tile-overlap 0.5
  "Overlap between neighbouring tiles, from 0 to 1."
  :type 'number)

(defcustom histopipe-max-background 0.5
  "Largest fraction of background a kept tile may have."
  :type 'number)

(defcustom histopipe-model "prostate_small"
  "HistoEncoder model."
  :type '(choice (const "prostate_small") (const "prostate_medium")))

(defcustom histopipe-num-blocks 1
  "Number of last attention blocks whose outputs make up the features."
  :type 'integer)

(defcustom histopipe-avg-pool nil
  "Non-nil appends the average-pooled patch tokens to the features."
  :type 'boolean)

(defcustom histopipe-batch-size 16
  "Batch size for feature extraction."
  :type 'integer)

(defcustom histopipe-workers 4
  "Data loading processes.  Using every core can hang on Windows."
  :type 'integer)

(defcustom histopipe-use-cpu nil
  "Non-nil extracts features on the CPU even when a GPU is found."
  :type 'boolean)

(defcustom histopipe-clusters '(8 16 32 64)
  "Numbers of clusters; each becomes a column of clusters.parquet."
  :type '(repeat integer))

(defcustom histopipe-view-column "n_clusters=8"
  "Clustering shown in cluster_views, a column of clusters.parquet."
  :type 'string)

(defcustom histopipe-view-tiles-per-cluster 16
  "Example tiles in each cluster_<k>.jpg grid."
  :type 'integer)

(defcustom histopipe-view-tile-px 128
  "Size in pixels of each example tile in the grids."
  :type 'integer)

(defcustom histopipe-views-after-cluster t
  "Non-nil makes cluster_views for every slide after the cluster step."
  :type 'boolean)

(defcustom histopipe-view-script nil
  "Your own script to make cluster_views instead of the built-in one.
It is called as SCRIPT SLIDE-FOLDER --slide SLIDE-FILE --column
COLUMN.  Leave it nil to use the built-in views, which also show
cluster names."
  :type '(choice (const :tag "Built-in" nil) file))

(defcustom histopipe-kaggle-competition "prostate-cancer-grade-assessment"
  "Kaggle competition that slides are downloaded from (PANDA)."
  :type 'string)

(defcustom histopipe-masks-directory "~/histo/masks"
  "Folder for downloaded label masks (<id>_mask.tiff)."
  :type 'directory)

(defcustom histopipe-train-csv "~/histo/train.csv"
  "Where the competition's train.csv is kept."
  :type 'file)

(defcustom histopipe-download-masks t
  "Non-nil downloads each slide's label mask along with it."
  :type 'boolean)

(defconst histopipe-steps '(tile extract cluster)
  "Pipeline steps, in order.")

(defvar histopipe--process nil
  "The running pipeline process, if any.")

(defvar histopipe--queue nil
  "Steps still to run after the current one.")

(defvar histopipe--overwrite nil
  "Non-nil while a run redoes slides that are already done.")

(defvar histopipe--finish #'ignore
  "Function called when the queue has run to the end.")

;;;; Paths

(defun histopipe--tiles () (file-name-as-directory (expand-file-name histopipe-tiles-directory)))
(defun histopipe--slides () (file-name-as-directory (expand-file-name histopipe-slides-directory)))

(defun histopipe--executable (name)
  "Return the full path of executable NAME, preferring `histopipe-venv'."
  (or (and histopipe-venv
           (cl-some (lambda (dir)
                      (cl-some (lambda (file)
                                 (let ((path (expand-file-name file (expand-file-name dir histopipe-venv))))
                                   (and (file-regular-p path) (file-executable-p path) path)))
                               (list (concat name ".exe") name)))
                    '("Scripts" "bin")))
      (executable-find name)
      (user-error "Cannot find %s in %s or on exec-path" name (or histopipe-venv "the venv"))))

(defun histopipe--slide-files ()
  "Slide files matching `histopipe-slide-pattern'."
  (cl-remove-if-not #'file-regular-p
                    (file-expand-wildcards (concat (histopipe--slides) histopipe-slide-pattern) t)))

(defun histopipe--slide-dir (slide)
  "Output folder of SLIDE, a slide file name."
  (expand-file-name (file-name-base slide) (histopipe--tiles)))

(defun histopipe--tiled-p (dir)
  "Non-nil when HistoPrep finished slide folder DIR."
  (or (file-exists-p (expand-file-name "metadata.parquet" dir))
      (file-exists-p (expand-file-name "metadata.csv" dir))))

(defun histopipe--slide-dirs ()
  "Slide folders under the tiles folder that contain tile images."
  (when (file-directory-p (histopipe--tiles))
    (cl-remove-if-not
     (lambda (dir) (file-directory-p (expand-file-name "tiles" dir)))
     (directory-files (histopipe--tiles) t "\\`[^.]"))))

(defun histopipe--has (dir file)
  (file-exists-p (expand-file-name file dir)))

(defun histopipe--number (n)
  "N as a string with a dot as decimal separator."
  (if (integerp n) (number-to-string n) (format "%s" n)))

;;;; Commands for each step

(defconst histopipe--histoprep-script
  "import sys; from histoprep._cli import cut_slides; cut_slides.main(args=sys.argv[1:], prog_name='HistoPrep')"
  "Runs HistoPrep with explicit arguments.
HistoPrep.exe would let click expand the *.tiff wildcard on
Windows, turning one --input pattern into many arguments.")

(defun histopipe--tile-command ()
  "Return (DESCRIPTION . ARGV) for the tile step, or a string why it is skipped."
  (let* ((slides (histopipe--slide-files))
         (pending (cl-remove-if (lambda (s) (and (not histopipe--overwrite)
                                                 (histopipe--tiled-p (histopipe--slide-dir s))))
                                slides)))
    (cond
     ((null slides)
      (user-error "No slides match %s in %s" histopipe-slide-pattern (histopipe--slides)))
     ((null pending) (format "all %d slides already tiled, skipping" (length slides)))
     (t (cons (format "Tiling %d of %d slides" (length pending) (length slides))
              (append
               (list (histopipe--executable "python") "-c" histopipe--histoprep-script
                     "--input" (concat (histopipe--slides) histopipe-slide-pattern)
                     "--output" (directory-file-name (histopipe--tiles))
                     "--level" (histopipe--number histopipe-level)
                     "--width" (histopipe--number histopipe-tile-width)
                     "--overlap" (histopipe--number histopipe-tile-overlap)
                     "--max-background" (histopipe--number histopipe-max-background)
                     "--num-workers" (histopipe--number histopipe-workers)
                     "--unfinished")
               (and histopipe--overwrite '("--overwrite"))))))))

(defun histopipe--extract-command ()
  "Return (DESCRIPTION . ARGV) for the extract step, or a string why it is skipped."
  (let* ((dirs (histopipe--slide-dirs))
         (pending (cl-remove-if (lambda (d) (and (not histopipe--overwrite)
                                                 (histopipe--has d "features.parquet")))
                                dirs)))
    (cond
     ((null dirs) (user-error "No tiled slides in %s; run the tile step first" (histopipe--tiles)))
     ((null pending) (format "features exist for all %d slides, skipping" (length dirs)))
     (t (cons (format "Extracting features for %d of %d slides" (length pending) (length dirs))
              (append
               (list (histopipe--executable "HistoEncoder") "extract"
                     "--input_dir" (directory-file-name (histopipe--tiles))
                     "--model-name" histopipe-model
                     "--num-blocks" (histopipe--number histopipe-num-blocks)
                     "--batch-size" (histopipe--number histopipe-batch-size)
                     "--num-workers" (histopipe--number histopipe-workers))
               (and histopipe-avg-pool '("--avg-pool"))
               (and histopipe-use-cpu '("--use-cpu"))
               (and histopipe--overwrite '("--overwrite"))))))))

(defun histopipe--cluster-command ()
  "Return (DESCRIPTION . ARGV) for the cluster step."
  (let ((done (cl-remove-if-not (lambda (d) (histopipe--has d "features.parquet"))
                                (histopipe--slide-dirs))))
    (unless done
      (user-error "No features.parquet in %s; run the extract step first" (histopipe--tiles)))
    (when histopipe-views-after-cluster
      ;; Run after this step, before anything queued later.
      (setq histopipe--queue
            (append (mapcar (lambda (d) (histopipe--view-step d histopipe-view-column)) done)
                    histopipe--queue)))
    (cons (format "Clustering %d slides into %s clusters" (length done)
                  (mapconcat #'number-to-string histopipe-clusters ", "))
          (append
           (list (histopipe--executable "HistoEncoder") "cluster"
                 "--input_dir" (directory-file-name (histopipe--tiles))
                 "--overwrite")
           (cl-mapcan (lambda (n) (list "--num-clusters" (number-to-string n)))
                      histopipe-clusters)))))

(defun histopipe--command (step)
  "Return (DESCRIPTION . ARGV) for queue item STEP, a skip message or :done."
  (pcase step
    (`(,_ ,function . ,_) (funcall function))
    ('tile (histopipe--tile-command))
    ('extract (histopipe--extract-command))
    ('cluster (histopipe--cluster-command))))

;;;; Log buffer

(defvar histopipe-log-mode-map
  (let ((map (make-sparse-keymap)))
    (define-key map (kbd "k") #'histopipe-stop)
    (define-key map (kbd "s") #'histopipe-status)
    map)
  "Keymap for `histopipe-log-mode'.")

(define-derived-mode histopipe-log-mode special-mode "Histopipe"
  "Output of the histopipe pipeline.")

(defun histopipe--log-buffer ()
  (let ((buffer (get-buffer-create "*histopipe*")))
    (with-current-buffer buffer
      (unless (derived-mode-p 'histopipe-log-mode) (histopipe-log-mode)))
    buffer))

(defun histopipe--log (format-string &rest args)
  "Append a line made from FORMAT-STRING and ARGS to the log buffer."
  (with-current-buffer (histopipe--log-buffer)
    (let ((inhibit-read-only t))
      (save-excursion
        (goto-char (point-max))
        (unless (bolp) (insert "\n"))
        (insert (propertize (apply #'format format-string args) 'face 'bold) "\n")))))

(defun histopipe--filter (process string)
  "Insert STRING from PROCESS, letting carriage returns redraw progress bars."
  (when (buffer-live-p (process-buffer process))
    (with-current-buffer (process-buffer process)
      (let ((inhibit-read-only t)
            (follow (= (point) (process-mark process))))
        (save-excursion
          (goto-char (process-mark process))
          (let ((start (line-beginning-position)))
            (insert (ansi-color-apply string))
            (set-marker (process-mark process) (point))
            (let ((end (point-marker)))
              (goto-char start)
              (while (search-forward "\r" end t)
                (if (eq (char-after) ?\n)
                    (delete-char -1)
                  (delete-region (line-beginning-position) (point))))
              (set-marker end nil))))
        (when follow (goto-char (process-mark process)))))))

;;;; Running

(defun histopipe--step-name (step)
  "Name of queue item STEP for the log."
  (if (consp step) (car step) (symbol-name step)))

(defun histopipe--pipeline-summary ()
  (histopipe--log "Finished: %d of %d slide folders have clusters.parquet."
                  (cl-count-if (lambda (d) (histopipe--has d "clusters.parquet"))
                               (histopipe--slide-dirs))
                  (length (histopipe--slide-dirs))))

(defun histopipe--next ()
  "Start the next queued step.
A queue item is a step symbol from `histopipe-steps', or a list
\(NAME FUNCTION OPTIONAL) whose FUNCTION returns what
`histopipe--command' returns.  A failed OPTIONAL item does not stop
the queue."
  (if (null histopipe--queue)
      (progn
        (funcall histopipe--finish)
        (message "histopipe: finished"))
    (let* ((step (pop histopipe--queue))
           (name (histopipe--step-name step))
           (command (condition-case err (histopipe--command step)
                      (user-error
                       (histopipe--log "==> %s: %s" name (error-message-string err))
                       (setq histopipe--queue nil)
                       (message "histopipe: %s" (error-message-string err))
                       nil))))
      (cond
       ((null command))
       ((eq command :done) (histopipe--next))
       ((stringp command)
        (histopipe--log "==> %s: %s" name command)
        (histopipe--next))
       (t
        (histopipe--log "==> %s: %s" name (car command))
        (histopipe--log "%s" (mapconcat #'shell-quote-argument (cdr command) " "))
        (let ((process-environment (append '("PYTHONUTF8=1" "PYTHONIOENCODING=utf-8"
                                             "PYTHONUNBUFFERED=1")
                                           process-environment))
              (buffer (histopipe--log-buffer)))
          (setq histopipe--process
                (make-process
                 :name (format "histopipe-%s" name)
                 :buffer buffer
                 :command (cdr command)
                 :coding 'utf-8
                 :connection-type 'pipe
                 :noquery t
                 :filter #'histopipe--filter
                 :sentinel (histopipe--sentinel
                            name (and (consp step) (nth 2 step))
                            (float-time) (with-current-buffer buffer (point-max)))))
          (with-current-buffer buffer
            (set-marker (process-mark histopipe--process) (point-max)))))))))

(defun histopipe--output-has (process start string)
  "Non-nil when PROCESS printed STRING after buffer position START."
  (with-current-buffer (process-buffer process)
    (save-excursion
      (goto-char start)
      (search-forward string nil t))))

(defun histopipe--sentinel (name optional started start)
  "Return a sentinel for step NAME, started at float time STARTED.
Its output begins at START in the log buffer.  When OPTIONAL is
non-nil a failure does not stop the queue."
  (lambda (process _event)
    (unless (process-live-p process)
      (setq histopipe--process nil)
      (let ((code (process-exit-status process))
            (elapsed (round (- (float-time) started))))
        (cond
         ((and (eq (process-status process) 'exit) (zerop code))
          (histopipe--log "    %s done in %d:%02d:%02d" name
                          (/ elapsed 3600) (% (/ elapsed 60) 60) (% elapsed 60))
          (histopipe--next))
         ((and optional (eq (process-status process) 'exit))
          (histopipe--log "    %s failed (exit %s), continuing." name code)
          (histopipe--next))
         (t
          (setq histopipe--queue nil)
          (histopipe--log "    %s failed (%s %s)." name (process-status process) code)
          (when (and (equal name "cluster")
                     (histopipe--output-has process start "has no attribute 'groupby'"))
            (histopipe--log "    Your polars is too new for HistoEncoder: pip install \"polars<1\" in the venv."))
          (when (cl-some (lambda (s) (histopipe--output-has process start s))
                         '("401" "403" "Unauthorized" "Forbidden" "kaggle.json"))
            (histopipe--log "    Kaggle refused the download.  Put kaggle.json (Kaggle > Settings > API)
    in ~/.kaggle/ and accept the rules at https://www.kaggle.com/c/%s/rules"
                            histopipe-kaggle-competition))
          (message "histopipe: %s failed" name)))))))

(defun histopipe-run-steps (steps &optional overwrite finish)
  "Run pipeline STEPS in order, skipping work already done.
With OVERWRITE non-nil, redo tiling and extraction for every slide.
FINISH is called with no arguments when every step succeeded; it
defaults to logging how many slides have clusters."
  (when (process-live-p histopipe--process)
    (user-error "The pipeline is already running; stop it with `histopipe-stop'"))
  (setq histopipe--queue (copy-sequence steps)
        histopipe--overwrite overwrite
        histopipe--finish (or finish #'histopipe--pipeline-summary))
  (with-current-buffer (histopipe--log-buffer)
    (let ((inhibit-read-only t))
      (erase-buffer)))
  (display-buffer (histopipe--log-buffer))
  (histopipe--log "Slides: %s%s" (histopipe--slides) histopipe-slide-pattern)
  (histopipe--log "Tiles:  %s" (histopipe--tiles))
  (histopipe--next))

(defun histopipe--read-directories ()
  "Ask for the slides and tiles folders and remember them for this session."
  (setq histopipe-slides-directory
        (read-directory-name "Slides folder: " histopipe-slides-directory nil t)
        histopipe-tiles-directory
        (read-directory-name "Tiles folder: " histopipe-tiles-directory)))

;;;###autoload
(defun histopipe-run (&optional arg)
  "Run tile, extract and cluster.
With one prefix ARG, ask for the slides and tiles folders first.
With two, also redo slides that are already tiled or extracted."
  (interactive "p")
  (when (> arg 1) (histopipe--read-directories))
  (histopipe-run-steps histopipe-steps (> arg 4)))

;;;###autoload
(defun histopipe-tile (&optional overwrite)
  "Cut slides into tiles.  With prefix arg OVERWRITE, redo every slide."
  (interactive "P")
  (histopipe-run-steps '(tile) overwrite))

;;;###autoload
(defun histopipe-extract (&optional overwrite)
  "Extract features.  With prefix arg OVERWRITE, redo every slide."
  (interactive "P")
  (histopipe-run-steps '(extract) overwrite))

;;;###autoload
(defun histopipe-cluster (&optional clusters)
  "Cluster the features of all slides.
With a prefix argument, ask for the numbers of CLUSTERS."
  (interactive
   (list (and current-prefix-arg
              (mapcar #'string-to-number
                      (split-string (read-string "Numbers of clusters: "
                                                 (mapconcat #'number-to-string histopipe-clusters " "))
                                    "[ ,]+" t)))))
  (let ((histopipe-clusters (or clusters histopipe-clusters)))
    ;; The command list is built when the step starts, inside this let.
    (histopipe-run-steps '(cluster))))

;;;###autoload
(defun histopipe-stop ()
  "Stop the running step and the rest of the pipeline."
  (interactive)
  (setq histopipe--queue nil)
  (if (process-live-p histopipe--process)
      (progn (interrupt-process histopipe--process)
             (run-at-time 3 nil (lambda (p) (when (process-live-p p) (kill-process p)))
                          histopipe--process)
             (histopipe--log "Stopping..."))
    (message "histopipe: nothing is running")))

;;;; Kaggle downloads

(defun histopipe--masks () (file-name-as-directory (expand-file-name histopipe-masks-directory)))

(defun histopipe--slide-id (string)
  "Slide id in STRING, such as a file name, path or mask name."
  (let ((name (file-name-nondirectory (string-trim string))))
    (replace-regexp-in-string "\\(_mask\\)?\\(\\.tiff\\)?\\(\\.zip\\)?\\'" "" name)))

(defun histopipe--kaggle-step (remote dir &optional optional)
  "Queue item that downloads competition file REMOTE into DIR.
If Kaggle serves the file zipped, it is unpacked and the zip
removed.  With OPTIONAL non-nil, a failure does not stop the queue."
  (let* ((file (file-name-nondirectory remote))
         (local (expand-file-name file dir)))
    (list (format "download %s" file)
          (lambda ()
            (if (and (file-exists-p local) (not histopipe--overwrite))
                (format "%s already in %s, skipping" file dir)
              (make-directory dir t)
              ;; The unzip item runs right after this download.
              (push (list (format "unzip %s" file)
                          (lambda () (histopipe--unzip-command local)))
                    histopipe--queue)
              (cons (format "%s from Kaggle" remote)
                    (append (list (histopipe--executable "kaggle")
                                  "competitions" "download"
                                  "-f" remote "-p" (directory-file-name dir))
                            (and histopipe--overwrite '("-o"))
                            (list histopipe-kaggle-competition)))))
          optional)))

(defun histopipe--unzip-command (local)
  "Command that unpacks LOCAL.zip (or its .zip twin) and deletes the zip, or :done."
  (let ((zip (cl-find-if #'file-exists-p
                         (list (concat local ".zip")
                               (concat (file-name-sans-extension local) ".zip")))))
    (if (null zip)
        :done
      (cons (format "unpacking %s" (file-name-nondirectory zip))
            (list (histopipe--executable "python") "-c"
                  "import os, sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2]); os.remove(sys.argv[1])"
                  zip (directory-file-name (file-name-directory local)))))))

(defun histopipe--slide-steps (ids masks)
  "Queue items that download slides IDS, and their label masks if MASKS."
  (cl-mapcan
   (lambda (id)
     (cons (histopipe--kaggle-step (format "train_images/%s.tiff" id) (histopipe--slides))
           (and masks
                ;; Some slides have no mask; carry on without it.
                (list (histopipe--kaggle-step (format "train_label_masks/%s_mask.tiff" id)
                                              (histopipe--masks) t)))))
   ids))

(defun histopipe--download-summary ()
  (histopipe--log "Finished: %d slides in %s, %d masks in %s."
                  (length (histopipe--slide-files)) (histopipe--slides)
                  (length (and (file-directory-p (histopipe--masks))
                               (directory-files (histopipe--masks) nil "_mask\\.tiff\\'")))
                  (histopipe--masks)))

(defun histopipe--id-at-point ()
  (let ((thing (thing-at-point 'symbol t)))
    (and thing (string-match "[0-9a-f]\\{32\\}" thing) (match-string 0 thing))))

;;;###autoload
(defun histopipe-download-slides (ids &optional masks)
  "Download slides IDS of `histopipe-kaggle-competition' into the slides folder.
IDS are slide ids or file names such as
0005f7aaab2800f6170c399693a96917.tiff, separated by spaces, commas
or newlines.  Label masks are downloaded too when MASKS is non-nil;
interactively that is `histopipe-download-masks', inverted by a
prefix argument."
  (interactive
   (list (split-string (read-string "Slide ids: " (histopipe--id-at-point)) "[ ,;\n\t]+" t)
         (if current-prefix-arg (not histopipe-download-masks) histopipe-download-masks)))
  (let ((ids (delete-dups (mapcar #'histopipe--slide-id ids))))
    (unless ids (user-error "No slide ids given"))
    (histopipe-run-steps (histopipe--slide-steps ids masks) nil #'histopipe--download-summary)))

;;;###autoload
(defun histopipe-download-train-csv ()
  "Download train.csv (slide id, data provider, ISUP grade, Gleason score)."
  (interactive)
  (histopipe-run-steps
   (list (histopipe--kaggle-step "train.csv" (file-name-directory (expand-file-name histopipe-train-csv))))
   nil (lambda () (histopipe--log "Finished: %s" (expand-file-name histopipe-train-csv)))))

(defun histopipe--read-train-csv ()
  "Rows of `histopipe-train-csv' as alists keyed by column name."
  (with-temp-buffer
    (insert-file-contents (expand-file-name histopipe-train-csv))
    (let* ((lines (split-string (buffer-string) "\r?\n" t))
           (header (split-string (car lines) ",")))
      (mapcar (lambda (line) (cl-mapcar #'cons header (split-string line ",")))
              (cdr lines)))))

(defun histopipe--sample-ids (n provider)
  "Pick N not yet downloaded slide ids from PROVIDER, spread over ISUP grades."
  (let* ((have (mapcar #'file-name-base (histopipe--slide-files)))
         (rows (cl-remove-if-not
                (lambda (row) (and (equal (cdr (assoc "data_provider" row)) provider)
                                   (not (member (cdr (assoc "image_id" row)) have))))
                (histopipe--read-train-csv)))
         (by-grade (make-hash-table :test #'equal))
         picked)
    (dolist (row rows)
      (push (cdr (assoc "image_id" row))
            (gethash (cdr (assoc "isup_grade" row)) by-grade)))
    (let ((grades (sort (hash-table-keys by-grade) #'string<)))
      ;; Shuffle within each grade, then take one per grade in turn.
      (dolist (grade grades)
        (puthash grade (mapcar #'cdr (sort (mapcar (lambda (id) (cons (random) id))
                                                   (gethash grade by-grade))
                                           (lambda (a b) (< (car a) (car b)))))
                 by-grade))
      (while (and (< (length picked) n)
                  (cl-some (lambda (g) (gethash g by-grade)) grades))
        (dolist (grade grades)
          (when (and (< (length picked) n) (gethash grade by-grade))
            (push (pop (gethash grade by-grade)) picked)))))
    (nreverse picked)))

;;;###autoload
(defun histopipe-download-sample (n &optional provider masks)
  "Download N slides from PROVIDER not downloaded yet, spread over ISUP grades.
PROVIDER is \"radboud\" (masks with Gleason 3/4/5) or \"karolinska\"
\(masks with benign/cancer only).  train.csv is downloaded first if
missing.  MASKS is as in `histopipe-download-slides'."
  (interactive
   (list (read-number "Number of slides: " 20)
         (completing-read "Data provider: " '("radboud" "karolinska") nil t nil nil "radboud")
         (if current-prefix-arg (not histopipe-download-masks) histopipe-download-masks)))
  (let ((provider (or provider "radboud")))
    (histopipe-run-steps
     (list (histopipe--kaggle-step "train.csv" (file-name-directory (expand-file-name histopipe-train-csv)))
           (list (format "pick %d %s slides" n provider)
                 (lambda ()
                   (let ((ids (histopipe--sample-ids n provider)))
                     (setq histopipe--queue (append (histopipe--slide-steps ids masks)
                                                    histopipe--queue))
                     (format "%d slides: %s" (length ids) (string-join ids " "))))))
     nil #'histopipe--download-summary)))

;;;; Status

(defvar histopipe-status-mode-map
  (let ((map (make-sparse-keymap)))
    (define-key map (kbd "RET") #'histopipe-status-open)
    (define-key map (kbd "t") #'histopipe-status-thumbnail)
    (define-key map (kbd "v") #'histopipe-view-slide)
    (define-key map (kbd "x") #'histopipe-slide-to-csv)
    (define-key map (kbd "n") #'histopipe-name-clusters)
    (define-key map (kbd "p") #'histopipe-pnd-predict-slide)
    (define-key map (kbd "P") #'histopipe-pnd-results)
    (define-key map (kbd "r") #'histopipe-run)
    (define-key map (kbd "k") #'histopipe-stop)
    (define-key map (kbd "l") (lambda () (interactive) (pop-to-buffer (histopipe--log-buffer))))
    map)
  "Keymap for `histopipe-status-mode'.")

(define-derived-mode histopipe-status-mode tabulated-list-mode "Histopipe-Status"
  "Slides and how far the pipeline got with each."
  (setq tabulated-list-format [("Slide" 36 t) ("Tiles" 8 histopipe--sort-tiles :right-align t)
                               ("Features" 9 t) ("Clusters" 9 t) ("PND" 4 t) ("True" 4 t)])
  (setq tabulated-list-sort-key '("Slide"))
  (add-hook 'tabulated-list-revert-hook #'histopipe--status-entries nil t)
  (tabulated-list-init-header))

(defun histopipe--sort-tiles (a b)
  (< (string-to-number (aref (cadr a) 1)) (string-to-number (aref (cadr b) 1))))

(defun histopipe--status-entries ()
  (let* ((dirs (histopipe--slide-dirs))
         (pnd (histopipe--pnd-results))
         (names (delete-dups
                 (append (mapcar #'file-name-base (histopipe--slide-files))
                         (mapcar #'file-name-nondirectory dirs)))))
    (setq tabulated-list-entries
          (mapcar
           (lambda (name)
             (let* ((dir (expand-file-name name (histopipe--tiles)))
                    (tile-dir (expand-file-name "tiles" dir))
                    (tiles (if (file-directory-p tile-dir)
                               (length (directory-files tile-dir nil "\\`[^.]" t))
                             0)))
               (list name
                     (vector name
                             (if (histopipe--tiled-p dir) (number-to-string tiles) "-")
                             (if (histopipe--has dir "features.parquet") "yes" "-")
                             (if (histopipe--has dir "clusters.parquet") "yes" "-")
                             (or (cdr (assoc "pnd_isup" (gethash name pnd))) "-")
                             (let ((true (cdr (assoc "isup_grade" (gethash name pnd)))))
                               (if (member true '(nil "")) "-" true))))))
           names))))

;;;###autoload
(defun histopipe-status ()
  "List slides with the pipeline steps done for each."
  (interactive)
  (with-current-buffer (get-buffer-create "*histopipe-status*")
    (histopipe-status-mode)
    (histopipe--status-entries)
    (tabulated-list-print t)
    (pop-to-buffer (current-buffer))))

(defun histopipe--read-slide ()
  "The slide on the current status line, or one read from the minibuffer."
  (or (and (derived-mode-p 'histopipe-status-mode) (tabulated-list-get-id))
      (completing-read "Slide: " (mapcar #'file-name-nondirectory (histopipe--slide-dirs)) nil t)))

(defun histopipe--output-dir (slide &optional needed)
  "Output folder of SLIDE.
Signal a clear error if it, or file NEEDED in it, is missing."
  (let ((dir (expand-file-name slide (histopipe--tiles))))
    (cond
     ((not (file-directory-p dir))
      (user-error "%s is not tiled yet; run the pipeline with `C-c h r' first" slide))
     ((and needed (not (file-exists-p (expand-file-name needed dir))))
      (user-error "%s has no %s yet; run the pipeline with `C-c h r' first" slide needed))
     (t dir))))

(defun histopipe-status-open (slide)
  "Open the output folder of SLIDE in Dired."
  (interactive (list (histopipe--read-slide)))
  (dired (histopipe--output-dir slide)))

(defun histopipe-status-thumbnail (slide)
  "Show the thumbnail of SLIDE with its tiles drawn on it."
  (interactive (list (histopipe--read-slide)))
  (let ((dir (histopipe--output-dir slide)))
    (find-file-other-window
     (cl-find-if #'file-exists-p
                 (mapcar (lambda (f) (expand-file-name f dir))
                         '("thumbnail_tiles.jpeg" "thumbnail.jpeg"))))))

;;;; Cluster views

(defconst histopipe--views-script "import csv, math, os, random, sys
import polars as pl
from PIL import Image, ImageDraw

slide_dir, slide_file, column, per_cluster, tile_px, names_file = sys.argv[1:7]
per_cluster, tile_px = int(per_cluster), int(tile_px)
COLORS = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0',
          '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8',
          '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080']

df = pl.read_parquet(os.path.join(slide_dir, 'clusters.parquet'))
columns = [c for c in df.columns if c.startswith('n_clusters')]
if column not in columns:
    sys.exit(f'No column {column} in clusters.parquet; choose from {columns}')
ids = df[column].to_list()
paths = df['path'].to_list()
boxes = list(zip(*(df[c].to_list() for c in 'xywh')))
clusters = sorted(set(ids))

names = {}
if os.path.exists(names_file):
    with open(names_file, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row['column'] == column and row['name'].strip():
                names[int(row['cluster'])] = row['name'].strip()
def label(k):
    return f'{k} {names[k]}' if k in names else f'cluster {k}'

out_dir = os.path.join(slide_dir, 'cluster_views')
os.makedirs(out_dir, exist_ok=True)
for f in os.listdir(out_dir):
    if f.startswith('cluster_') and f.endswith('.jpg'):
        os.remove(os.path.join(out_dir, f))

counts = {k: ids.count(k) for k in clusters}
with open(os.path.join(out_dir, 'summary.csv'), 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['cluster', 'name', 'tiles', 'percent'])
    for k in sorted(clusters, key=lambda k: -counts[k]):
        w.writerow([k, names.get(k, ''), counts[k], round(100 * counts[k] / len(ids), 1)])
        print(f'cluster {k:>3}  {counts[k]:>7} tiles  {100 * counts[k] / len(ids):5.1f}%  {names.get(k, \"\")}')

def tile_file(p):
    if os.path.exists(p):
        return p
    alt = os.path.join(slide_dir, 'tiles', os.path.basename(p.replace('\\\\', '/')))
    return alt if os.path.exists(alt) else None

random.seed(0)
for k in clusters:
    files = [f for f in (tile_file(p) for p, c in zip(paths, ids) if c == k) if f]
    if not files:
        print(f'cluster {k}: tile images not found, skipped')
        continue
    files = random.sample(files, min(per_cluster, len(files)))
    cols = max(1, math.ceil(math.sqrt(len(files))))
    rows = math.ceil(len(files) / cols)
    grid = Image.new('RGB', (cols * tile_px, rows * tile_px + 24), 'white')
    for i, f in enumerate(files):
        with Image.open(f) as im:
            grid.paste(im.convert('RGB').resize((tile_px, tile_px)),
                       ((i % cols) * tile_px, 24 + (i // cols) * tile_px))
    ImageDraw.Draw(grid).text((6, 6), f'{label(k)}: {counts[k]} tiles', fill='black')
    grid.save(os.path.join(out_dir, f'cluster_{k}.jpg'), quality=90)

thumb_file = os.path.join(slide_dir, 'thumbnail.jpeg')
if not os.path.exists(thumb_file):
    print('No thumbnail.jpeg, skipping overlay')
    sys.exit(0)
thumb = Image.open(thumb_file).convert('RGBA')
width = None
if slide_file:
    try:
        from histoprep import SlideReader
        width = SlideReader(slide_file).dimensions[1]
    except Exception as e:
        print(f'Could not read slide size ({e}); estimating it')
if not width:
    # The thumbnail is a pyramid level: assume the next power of two.
    ratio = max(x + w for x, y, w, h in boxes) / thumb.width
    width = thumb.width * 2 ** math.ceil(math.log2(max(ratio, 1)))
scale = thumb.width / width
layer = Image.new('RGBA', thumb.size, (0, 0, 0, 0))
draw = ImageDraw.Draw(layer)
color = {k: COLORS[i % len(COLORS)] for i, k in enumerate(clusters)}
def rgba(h, a):
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5)) + (a,)
for (x, y, w, h), k in zip(boxes, ids):
    draw.rectangle([x * scale, y * scale, (x + w) * scale, (y + h) * scale], fill=rgba(color[k], 90))
overlay = Image.alpha_composite(thumb, layer)
legend = 30 + 7 * max(len(label(k)) for k in clusters)
canvas = Image.new('RGB', (overlay.width + legend, max(overlay.height, 20 * len(clusters) + 10)), 'white')
canvas.paste(overlay.convert('RGB'), (0, 0))
d = ImageDraw.Draw(canvas)
for i, k in enumerate(clusters):
    y0 = 10 + i * 20
    d.rectangle([overlay.width + 10, y0, overlay.width + 24, y0 + 14], fill=color[k])
    d.text((overlay.width + 30, y0), label(k), fill='black')
canvas.save(os.path.join(out_dir, 'overlay.jpg'), quality=90)
print(f'Wrote {out_dir}')
"
  "Python that writes cluster_views for one slide.
Arguments: SLIDE-FOLDER SLIDE-FILE COLUMN TILES-PER-CLUSTER TILE-PX
NAMES-CSV.  Only uses polars calls that exist in old and new versions.")

(defun histopipe--names-file ()
  "CSV with the names given to clusters: column, cluster, name."
  (expand-file-name "cluster_names.csv" (histopipe--tiles)))

(defun histopipe--slide-file (slide)
  "The slide image of SLIDE, or nil."
  (cl-find-if (lambda (f) (string= (file-name-base f) slide)) (histopipe--slide-files)))

(defun histopipe--view-command (dir column)
  "Command that writes DIR/cluster_views for clustering COLUMN."
  (let* ((slide (file-name-nondirectory (directory-file-name dir)))
         (file (histopipe--slide-file slide)))
    (unless (file-exists-p (expand-file-name "clusters.parquet" dir))
      (user-error "%s has no clusters.parquet yet; run the pipeline with `C-c h r' first" slide))
    (cons (format "cluster_views for %s (%s)" slide column)
          (if histopipe-view-script
              (append (list (histopipe--executable "python")
                            (expand-file-name histopipe-view-script) dir)
                      (and file (list "--slide" file))
                      (list "--column" column))
            (list (histopipe--executable "python") "-c" histopipe--views-script
                  dir (or file "") column
                  (number-to-string histopipe-view-tiles-per-cluster)
                  (number-to-string histopipe-view-tile-px)
                  (histopipe--names-file))))))

(defun histopipe--view-step (dir column)
  "Queue item that makes the cluster views of slide folder DIR.
A failure does not stop the queue."
  (list (format "views %s" (file-name-nondirectory (directory-file-name dir)))
        (lambda () (histopipe--view-command dir column))
        t))

(defun histopipe--read-column ()
  (completing-read "Clustering: "
                   (mapcar (lambda (n) (format "n_clusters=%d" n)) histopipe-clusters)
                   nil nil nil nil histopipe-view-column))

;;;###autoload
(defun histopipe-view-slide (slide &optional column)
  "Write SLIDE's cluster_views folder and open it.
It holds cluster_<k>.jpg (example tiles of each cluster),
overlay.jpg (the thumbnail coloured by cluster) and summary.csv
\(tiles and share of the slide per cluster, with your names).
COLUMN is the clustering, `histopipe-view-column' unless a prefix
argument asks for it."
  (interactive (list (histopipe--read-slide)
                     (and current-prefix-arg (histopipe--read-column))))
  (let* ((dir (histopipe--output-dir slide "clusters.parquet"))
         (views (expand-file-name "cluster_views" dir)))
    (histopipe-run-steps
     (list (histopipe--view-step dir (or column histopipe-view-column)))
     nil (lambda ()
           (when (file-directory-p views) (dired-other-window views))))))

;;;###autoload
(defun histopipe-view-all (&optional column)
  "Write cluster_views for every clustered slide.
With a prefix argument, ask for the clustering COLUMN."
  (interactive (list (and current-prefix-arg (histopipe--read-column))))
  (let ((dirs (cl-remove-if-not (lambda (d) (histopipe--has d "clusters.parquet"))
                                (histopipe--slide-dirs))))
    (unless dirs (user-error "No clustered slides yet; run the pipeline with `C-c h r' first"))
    (histopipe-run-steps
     (mapcar (lambda (d) (histopipe--view-step d (or column histopipe-view-column))) dirs)
     nil (lambda () (histopipe--log "Finished: cluster_views for %d slides." (length dirs))))))

;;;; Cluster names

(defun histopipe--csv-fields (line)
  "Split CSV LINE into fields, honouring double quotes."
  (let ((fields nil) (field "") (i 0) (quoted nil) (n (length line)))
    (while (< i n)
      (let ((c (aref line i)))
        (cond
         ((and quoted (eq c ?\") (< (1+ i) n) (eq (aref line (1+ i)) ?\"))
          (setq field (concat field "\"") i (1+ i)))
         ((eq c ?\") (setq quoted (not quoted)))
         ((and (eq c ?,) (not quoted)) (push field fields) (setq field ""))
         (t (setq field (concat field (string c))))))
      (setq i (1+ i)))
    (nreverse (cons field fields))))

(defun histopipe--csv-field (string)
  (if (string-match-p "[\",\n]" string)
      (concat "\"" (replace-regexp-in-string "\"" "\"\"" string) "\"")
    string))

(defun histopipe--read-names ()
  "Cluster names as a list of (COLUMN CLUSTER NAME)."
  (let ((file (histopipe--names-file)))
    (when (file-exists-p file)
      (with-temp-buffer
        (insert-file-contents file)
        (cl-loop for line in (cdr (split-string (buffer-string) "\r?\n" t))
                 for (column cluster name) = (histopipe--csv-fields line)
                 when (and cluster name)
                 collect (list column (string-to-number cluster) name))))))

(defun histopipe--write-names (names)
  "Save NAMES, a list of (COLUMN CLUSTER NAME), to the names file."
  (with-temp-file (histopipe--names-file)
    (insert "column,cluster,name\n")
    (dolist (entry (sort (copy-sequence names)
                         (lambda (a b) (if (equal (car a) (car b))
                                           (< (cadr a) (cadr b))
                                         (string< (car a) (car b))))))
      (insert (mapconcat #'histopipe--csv-field
                         (list (nth 0 entry) (number-to-string (nth 1 entry)) (nth 2 entry))
                         ",")
              "\n"))))

;;;###autoload
(defun histopipe-name-clusters (slide &optional column)
  "Show SLIDE's example grid of each cluster and ask for a name.
Names such as \"benign glands\", \"stroma\", \"tumour glands\" or
\"edge/fold artifact\" are kept in cluster_names.csv in the tiles
folder and apply to every slide, since clusters are shared.  RET
keeps a name, \"-\" removes it.  The slide's views are then redone.
Re-clustering after adding slides renumbers the clusters, so name
them again then."
  (interactive (list (histopipe--read-slide)
                     (and current-prefix-arg (histopipe--read-column))))
  (let* ((column (or column histopipe-view-column))
         (views (expand-file-name "cluster_views" (histopipe--output-dir slide "clusters.parquet")))
         (grids (and (file-directory-p views)
                     (sort (directory-files views t "\\`cluster_[0-9]+\\.jpg\\'")
                           (lambda (a b) (< (histopipe--grid-number a) (histopipe--grid-number b))))))
         (names (histopipe--read-names))
         (window (selected-window)))
    (unless grids
      (user-error "No cluster grids for %s yet; make them with `v' first" slide))
    (unwind-protect
        (dolist (grid grids)
          (let* ((k (histopipe--grid-number grid))
                 (old (nth 2 (cl-find-if (lambda (e) (and (equal (car e) column) (= (cadr e) k)))
                                         names)))
                 ;; Without image support (a terminal) the prompt names the file instead.
                 (buffer (and (display-images-p) (find-file-noselect grid))))
            (when buffer
              (display-buffer buffer '(display-buffer-use-some-window (inhibit-same-window . t))))
            (let ((name (string-trim (read-string (format "Cluster %d of %s%s (RET keeps%s): "
                                                          k column
                                                          (if buffer "" (format " [%s]" grid))
                                                          (if old (format " \"%s\"" old) ""))
                                                  nil nil old))))
              (setq names (cl-remove-if (lambda (e) (and (equal (car e) column) (= (cadr e) k)))
                                        names))
              (unless (member name '("" "-"))
                (push (list column k name) names)))
            (when buffer (kill-buffer buffer))))
      (histopipe--write-names names)
      (select-window window))
    (message "histopipe: saved names in %s" (histopipe--names-file))
    (histopipe-view-slide slide column)))

(defun histopipe--grid-number (file)
  (string-to-number (replace-regexp-in-string "\\`cluster_\\|\\.jpg\\'" "" (file-name-nondirectory file))))

;;;; Parquet to CSV

(defconst histopipe--csv-script
  "import os, sys, polars as pl
digits = int(sys.argv[1])
for src, dst in zip(sys.argv[2::2], sys.argv[3::2]):
    df = pl.read_parquet(src)
    if 'path' in df.columns:
        df = df.with_columns(pl.Series('tile', [os.path.basename(str(p)) for p in df['path']]))
        df = df.select(['tile'] + [c for c in df.columns if c not in ('tile', 'path')])
    df = df.with_columns([pl.col(c).round(digits) for c, t in zip(df.columns, df.dtypes)
                          if t in (pl.Float32, pl.Float64)])
    df.write_csv(dst)
    print(f'{dst}: {df.height} rows, {df.width} columns')
"
  "Python that writes each parquet file as CSV.
Uses only polars calls that exist in old and new versions.")

(defcustom histopipe-csv-decimals 4
  "Decimals kept for numbers with a fraction, such as features."
  :type 'integer)

(defun histopipe--csv-name (parquet)
  (concat (file-name-sans-extension parquet) ".csv"))

;;;###autoload
(defun histopipe-parquet-to-csv (files)
  "Write each parquet file in FILES as a CSV file next to it.
The tile path column becomes the tile file name, first, and
fractions are rounded to `histopipe-csv-decimals'.  In Dired the
marked files are used, otherwise a file is read from the
minibuffer.  A single result is opened when done."
  (interactive
   (list (or (and (derived-mode-p 'dired-mode)
                  (cl-remove-if-not (lambda (f) (string-suffix-p ".parquet" f))
                                    (dired-get-marked-files)))
             (list (read-file-name "Parquet file: " nil nil t nil
                                   (lambda (f) (or (file-directory-p f)
                                                   (string-suffix-p ".parquet" f))))))))
  (let ((files (mapcar #'expand-file-name files))
        (buffer (get-buffer-create "*histopipe-csv*"))
        (process-environment (cons "PYTHONUTF8=1" process-environment)))
    (unless files (user-error "No parquet files"))
    (with-current-buffer buffer (let ((inhibit-read-only t)) (erase-buffer)))
    (make-process
     :name "histopipe-csv"
     :buffer buffer
     :command (append (list (histopipe--executable "python") "-c" histopipe--csv-script
                            (number-to-string histopipe-csv-decimals))
                      (cl-mapcan (lambda (f) (list f (histopipe--csv-name f))) files))
     :sentinel (lambda (process _event)
                 (unless (process-live-p process)
                   (cond
                    ((not (zerop (process-exit-status process)))
                     (pop-to-buffer (process-buffer process))
                     (message "histopipe: CSV conversion failed"))
                    ((= (length files) 1)
                     (find-file-other-window (histopipe--csv-name (car files))))
                    (t (message "histopipe: wrote %d CSV files" (length files)))))))
    (message "histopipe: converting %d parquet file%s"
             (length files) (if (cdr files) "s" ""))))

;;;###autoload
(defun histopipe-slide-to-csv (slide)
  "Write every parquet file of SLIDE's output folder as CSV."
  (interactive (list (histopipe--read-slide)))
  (let ((files (directory-files (histopipe--output-dir slide) t "\\.parquet\\'")))
    (unless files (user-error "No parquet files for %s" slide))
    (histopipe-parquet-to-csv files)))

;;;; PANDA 1st-place ensemble

(defconst histopipe--directory
  (file-name-directory (or load-file-name buffer-file-name default-directory))
  "Folder histopipe.el was loaded from.")

(defcustom histopipe-pnd-script nil
  "pnd_predict.py, which runs the PANDA 1st-place ensemble.
nil looks for pnd/pnd_predict.py next to histopipe.el, then for
pnd_predict.py in the folder above the tiles folder."
  :type '(choice (const :tag "Find it" nil) file))

(defcustom histopipe-pnd-directory nil
  "Your copy of Kaggle-PANDA-1st-place-solution, with the weights.
nil means Kaggle-PANDA-1st-place-solution in the folder above the
tiles folder.  It is cloned there with git when it is missing."
  :type '(choice (const :tag "Next to the tiles folder" nil) directory))

(defcustom histopipe-pnd-holdout t
  "Non-nil grades each slide only with the models that never trained on it.
Keep it on for PANDA train slides, or the scores look better than
they are; turn it off for slides from elsewhere to use every model."
  :type 'boolean)

(defcustom histopipe-pnd-speed 'full
  "How much of the ensemble runs for each slide."
  :type '(choice (const :tag "Every model with 12-view TTA (minutes per slide)" full)
                 (const :tag "Every model, no TTA (under a minute)" no-tta)
                 (const :tag "One b1 and one b0 model, no TTA (seconds)" fast)))

(defcustom histopipe-pnd-threads nil
  "CPU threads for torch in pnd_predict.py; nil lets torch decide."
  :type '(choice (const :tag "torch default" nil) integer))

(defun histopipe--parent () (file-name-directory (directory-file-name (histopipe--tiles))))

(defun histopipe--pnd-csv () (expand-file-name "pnd_slides.csv" (histopipe--tiles)))

(defun histopipe--pnd-dir ()
  (file-name-as-directory
   (expand-file-name (or histopipe-pnd-directory
                         (expand-file-name "Kaggle-PANDA-1st-place-solution" (histopipe--parent))))))

(defun histopipe--pnd-script ()
  "Full path of pnd_predict.py, or a user error saying where to put it."
  (or (cl-find-if #'file-exists-p
                  (if histopipe-pnd-script
                      (list (expand-file-name histopipe-pnd-script))
                    (list (expand-file-name "pnd/pnd_predict.py" histopipe--directory)
                          (expand-file-name "pnd_predict.py" (histopipe--parent)))))
      (user-error "Cannot find pnd_predict.py; put it in %s or set `histopipe-pnd-script'"
                  (histopipe--parent))))

(defun histopipe--pnd-clone-step ()
  "Queue item that clones the PND repository, with its weights, when missing."
  (let ((dir (histopipe--pnd-dir)))
    (list "PND weights"
          (lambda ()
            (if (file-directory-p (expand-file-name "final_models" dir))
                (format "found in %s" dir)
              (when (file-exists-p (directory-file-name dir))
                (user-error "%s has no final_models folder; delete it or set `histopipe-pnd-directory'"
                            dir))
              (cons "cloning Kaggle-PANDA-1st-place-solution (about 350 MB)"
                    (list (histopipe--executable "git") "clone" "--depth" "1"
                          "https://github.com/kentaroy47/Kaggle-PANDA-1st-place-solution"
                          (directory-file-name dir))))))))

(defun histopipe--pnd-command (slides redo)
  "Argument list of pnd_predict.py for SLIDES (files, or the slides folder)."
  (append (list (histopipe--executable "python") (histopipe--pnd-script))
          slides
          (list "--pnd-dir" (directory-file-name (histopipe--pnd-dir))
                "--out" (histopipe--pnd-csv)
                "--train-csv" (expand-file-name histopipe-train-csv)
                "--pattern" histopipe-slide-pattern)
          (and histopipe-pnd-holdout '("--holdout"))
          (pcase histopipe-pnd-speed ('no-tta '("--no-tta")) ('fast '("--fast")))
          (and histopipe-pnd-threads (list "--threads" (number-to-string histopipe-pnd-threads)))
          (and redo '("--redo"))))

(defun histopipe--pnd-results ()
  "Hash table of slide id -> alist of its row in pnd_slides.csv."
  (let ((table (make-hash-table :test #'equal))
        (file (histopipe--pnd-csv)))
    (when (file-exists-p file)
      (with-temp-buffer
        (insert-file-contents file)
        (let* ((lines (split-string (buffer-string) "\r?\n" t))
               (header (histopipe--csv-fields (car lines))))
          (dolist (line (cdr lines))
            (let ((row (cl-mapcar #'cons header (histopipe--csv-fields line))))
              (puthash (cdr (assoc "image_id" row)) row table))))))
    table))

(defun histopipe--pnd-summary ()
  (let ((rows (hash-table-values (histopipe--pnd-results))))
    (histopipe--log "Finished: %d slides graded in %s%s" (length rows) (histopipe--pnd-csv)
                    (let ((known (cl-remove-if (lambda (r) (equal (cdr (assoc "isup_grade" r)) ""))
                                               rows)))
                      (if (null known) ""
                        (format "; %d of %d match the true ISUP grade"
                                (cl-count-if (lambda (r) (equal (cdr (assoc "isup_grade" r))
                                                                (cdr (assoc "pnd_isup" r))))
                                             known)
                                (length known)))))
    (when (get-buffer "*histopipe-status*")
      (with-current-buffer "*histopipe-status*"
        (histopipe--status-entries)
        (tabulated-list-print t)))))

(defun histopipe--pnd-read-options ()
  "Ask for the ensemble options; return non-nil to redo graded slides."
  (setq histopipe-pnd-speed
        (intern (completing-read "Speed: " '("full" "no-tta" "fast") nil t nil nil
                                 (symbol-name histopipe-pnd-speed)))
        histopipe-pnd-holdout
        (y-or-n-p "Only models that never trained on each slide (for PANDA train slides)? "))
  (y-or-n-p "Redo slides that are already graded? "))

(defun histopipe--pnd-run (slides description redo)
  (let ((command (histopipe--pnd-command slides redo)))
    (histopipe-run-steps
     (list (histopipe--pnd-clone-step)
           (list "pnd" (lambda () (cons description command))))
     nil #'histopipe--pnd-summary)))

;;;###autoload
(defun histopipe-pnd-predict-all (&optional ask)
  "Grade every slide with the PANDA 1st-place ensemble (pnd_predict.py).
Results go to pnd_slides.csv in the tiles folder and to the PND and
True columns of the status list; graded slides are skipped, so a
stopped run carries on where it left off.  The options are
`histopipe-pnd-speed' and `histopipe-pnd-holdout'; with a prefix
argument ASK, they are asked for and kept for this session."
  (interactive "P")
  (let ((redo (and ask (histopipe--pnd-read-options))))
    (unless (histopipe--slide-files)
      (user-error "No %s slides in %s" histopipe-slide-pattern (histopipe--slides)))
    (histopipe--pnd-run (list (directory-file-name (histopipe--slides)))
                        (format "PANDA ensemble (%s%s) on %d slides" histopipe-pnd-speed
                                (if histopipe-pnd-holdout ", held-out models" "")
                                (length (histopipe--slide-files)))
                        redo)))

;;;###autoload
(defun histopipe-pnd-predict-slide (slide &optional ask)
  "Grade SLIDE with the PANDA 1st-place ensemble.
It is regraded if it was graded before.  With a prefix argument
ASK, the options are asked for first, as in `histopipe-pnd-predict-all'."
  (interactive
   (list (or (and (derived-mode-p 'histopipe-status-mode) (tabulated-list-get-id))
             (completing-read "Slide: " (mapcar #'file-name-base (histopipe--slide-files)) nil t))
         current-prefix-arg))
  (when ask (histopipe--pnd-read-options))
  (let ((file (or (histopipe--slide-file slide)
                  (user-error "No slide file for %s in %s" slide (histopipe--slides)))))
    (histopipe--pnd-run (list file)
                        (format "PANDA ensemble (%s%s) on %s" histopipe-pnd-speed
                                (if histopipe-pnd-holdout ", held-out models" "") slide)
                        t)))

;;;###autoload
(defun histopipe-pnd-results ()
  "Open pnd_slides.csv, the grades from the PANDA ensemble."
  (interactive)
  (unless (file-exists-p (histopipe--pnd-csv))
    (user-error "No grades yet; run `histopipe-pnd-predict-all' (C-c h p) first"))
  (find-file-other-window (histopipe--pnd-csv)))

;;;; Key map

;;;###autoload
(defvar histopipe-command-map
  (let ((map (make-sparse-keymap)))
    (define-key map (kbd "r") #'histopipe-run)
    (define-key map (kbd "t") #'histopipe-tile)
    (define-key map (kbd "e") #'histopipe-extract)
    (define-key map (kbd "c") #'histopipe-cluster)
    (define-key map (kbd "k") #'histopipe-stop)
    (define-key map (kbd "s") #'histopipe-status)
    (define-key map (kbd "v") #'histopipe-view-slide)
    (define-key map (kbd "d") #'histopipe-download-slides)
    (define-key map (kbd "D") #'histopipe-download-sample)
    (define-key map (kbd "x") #'histopipe-parquet-to-csv)
    (define-key map (kbd "n") #'histopipe-name-clusters)
    (define-key map (kbd "V") #'histopipe-view-all)
    (define-key map (kbd "p") #'histopipe-pnd-predict-all)
    (define-key map (kbd "P") #'histopipe-pnd-results)
    map)
  "Keymap for histopipe commands; bind it to a prefix key.")
;;;###autoload
(fset 'histopipe-command-map histopipe-command-map)

(provide 'histopipe)
;;; histopipe.el ends here
