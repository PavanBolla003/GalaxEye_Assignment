# Design note

This project is a compact offline satellite tile classification pipeline based on a small CNN trained in PyTorch. The current repository does not yet include a web service, database-backed API, or analyst dashboard; it implements the model-training and evaluation path that proves the core ML workflow runs locally on CPU.

## What is already implemented

The repo contains:

- `train_cnn.py` for loading a labelled image dataset, training the model, and saving a checkpoint.
- `test_cnn.py` for loading the saved checkpoint, running evaluation on a held-out or labeled set, and writing metrics and predictions.
- `models/basic_cnn.pth` as the trained model artifact.

The dataset used for this assignment is the provided satellite tile set, arranged into class folders for training and an evaluation set with ground-truth labels.

## End-to-end flow

```text
Assignment dataset
   |
   v
train_cnn.py
   |
   |-- reads class folders from candidate_tiles
   |-- applies resize + augmentation
   |-- trains BasicCNN
   |-- saves model checkpoint to models/basic_cnn.pth
   v
test_cnn.py
   |
   |-- loads checkpoint
   |-- reads eval_set and eval_labels.csv
   |-- runs inference on each tile
   |-- computes metrics, confusion matrix, and predictions.csv
```

The key design choice here is simplicity: the task is solved as a local offline training and inference workflow rather than as a production service. That makes it appropriate for CPU-only hardware and for isolated systems with no internet access.

## Model and data decisions

The classifier uses a straightforward CNN with:

- 64x64 RGB input resizing
- basic convolutional feature extraction
- batch normalization and ReLU
- adaptive pooling and a small dense classifier
- cross-entropy training

The dataset uses the standard `ImageFolder` layout from PyTorch, which works well for labelled folders such as Forest, River, Residential, Industrial, and so on.

For the assignment dataset, the chosen path is practical because it is:

- simple to train and debug,
- easy to evaluate on CPU,
- compatible with offline execution,
- and sufficient for a small proof of concept.

## Trade-offs considered

### 1. Simple local CNN instead of a larger pretrained model

Why this choice:

- fast to train,
- easy to reason about,
- no hosted API dependency,
- suitable for short offline experiments.

Trade-off:

- it may not match the performance of a more specialized or larger model,
- but it is enough to prove the classification pipeline works.

### 2. Confidence handling

The evaluation script already computes softmax probabilities and saves the maximum confidence per sample. That is useful for identifying uncertain predictions, but confidence is not the same as calibrated reliability.

In a production extension, low-confidence predictions would be flagged for human review rather than silently accepted. This is the right operational pattern for offline land-use classification, where mistakes can be costly.

### 3. Storage for results

The repo currently stores outputs as CSV and plots rather than a queryable database. For the current state of the project, this is enough because the goal is to prove the model pipeline works, not to build a full analyst-facing service.

A future version could store prediction results in SQLite or a local database, with fields such as:

- tile ID
- timestamp
- predicted class
- confidence
- model version
- ground truth if available

That would make analyst queries and auditing easier, while still keeping everything offline.

## Assumptions and limitations

- The model is trained on the downloaded tile dataset and is designed for the classes provided in the assignment.
- The repository is a model development and evaluation prototype, not a finished production service.
- No hosted APIs are used; inference is done locally on the machine where the code runs.
- The scripts assume the downloaded dataset is present in the expected folder structure.

## Future extension direction

If this were extended into the full offline service described in the assignment, the next step would be:

1. expose a local API endpoint that receives one tile at a time,
2. load the model once at startup,
3. classify the tile,
4. store predictions in a local database,
5. provide analyst queries for predicted class and confidence by time or tile ID.

That is the natural next layer beyond the current code, but it is not implemented in this repository yet.