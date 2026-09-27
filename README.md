# Satellite Tile Classifier

This project is a local PyTorch image-classification pipeline for the GalaxEye take-home dataset. It trains a CNN on labelled satellite tiles and evaluates it on a labelled evaluation set. The current version proves the model pipeline works locally and offline; it does not yet include a REST API or persistent database for analyst queries.

## What is in this repo

- `train_cnn.py` — trains the model on the data in the assignment dataset.
- `test_cnn.py` — loads the saved model, evaluates performance, and saves CSV/plot outputs.
- `models/basic_cnn.pth` — trained checkpoint.
- `DESIGN.md` — design note describing the approach and trade-offs.

## Project flow

1. Put the downloaded dataset folder in a location accessible from this repo.
2. Train the model by running `train_cnn.py`.
3. Evaluate the checkpoint with `test_cnn.py`.
4. Inspect the metrics and output files in `results/`.

## Folder structure needed

The scripts expect the dataset to be arranged like this:

```text
candidate_tiles/
    Forest/
        ...jpg files
    River/
        ...jpg files
    Residential/
        ...jpg files
    ...

eval_set/
    ...jpg files

eval_labels.csv
```

The dataset that you downloaded contains exactly this structure and includes the class names described in the assignment README.

## Install dependencies

Use a Python 3.10+ environment and install:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch torchvision numpy tqdm pandas scikit-learn matplotlib seaborn pillow
```

If you are running in an isolated environment without internet access, install from a local wheelhouse instead:

```powershell
python -m pip install --no-index --find-links D:\wheelhouse torch torchvision numpy tqdm pandas scikit-learn matplotlib seaborn pillow
```

## Train the model

From the project root, run:

```powershell
python train_cnn.py
```

This script:

- loads the dataset from `candidate_tiles`,
- resizes and augments the images,
- trains the CNN,
- and saves the trained model to `models/basic_cnn.pth`.

## Evaluate the model

Make sure the downloaded evaluation dataset is present and that `eval_labels.csv` is in the project root. Then run:

```powershell
python test_cnn.py
```

This script:

- loads the saved checkpoint,
- runs prediction on the evaluation tiles,
- computes metrics such as accuracy, precision, recall, and F1,
- saves a confusion matrix and prediction CSVs under `results/`.

## Output files produced

After evaluation, the script generates:

- `results/classification_report.txt`
- `results/confusion_matrix.png`
- `results/predictions.csv`
- `results/misclassified.csv`

## Notes

- This is a working offline ML prototype, not a hosted service.
- The model is simple and intentionally lightweight for local execution.
- The scripts are designed to run on CPU by default and only use CUDA when available.
- Windows users should be careful with DataLoader workers; if needed, set `NUM_WORKERS = 0` in the scripts for better compatibility.

## Current status

This repository proves the core pipeline:

- data ingestion,
- model training,
- checkpoint saving,
- model evaluation,
- and result output generation.

The next step for the full assignment would be wrapping this into a local inference API and storing predictions in an offline database for analyst queries, but that work is not yet implemented here.