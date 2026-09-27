import os
import time
import torch
import torch.nn as nn
import pandas as pd

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# CONFIG
# ============================================================

# Files and directories required to evaluate the trained model.
EVAL_DIR = "eval_set"
LABEL_FILE = "eval_labels.csv"
MODEL_FILE = "models/basic_cnn.pth"

IMG_SIZE = 64
BATCH_SIZE = 32
NUM_WORKERS = 0

RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# DEVICE
# ============================================================

# Run inference on GPU when available; otherwise fall back to CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("DEVICE")
print("=" * 70)
print(device)


# ============================================================
# TRANSFORM
# ============================================================

# Use the same input preprocessing used during training for valid evaluation.
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ============================================================
# CNN MODEL
# ============================================================

# Lightweight CNN used to extract tile features and classify each image.
class BasicCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ============================================================
# MAIN
# ============================================================

def main():
    # Load the saved training checkpoint and rebuild the model architecture.
    print("\n" + "=" * 70)
    print("LOADING MODEL")
    print("=" * 70)

    checkpoint = torch.load(MODEL_FILE, map_location=device)
    classes = checkpoint["classes"]
    num_classes = len(classes)

    print("Classes:")
    for i, cls in enumerate(classes):
        print(f"  {i}: {cls}")

    model = BasicCNN(num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    total_parameters = sum(p.numel() for p in model.parameters())
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = os.path.getsize(MODEL_FILE) / (1024 ** 2)

    print("\nModel parameters:")
    print(f"Total      : {total_parameters:,}")
    print(f"Trainable  : {trainable_parameters:,}")
    print(f"Model size : {model_size_mb:.2f} MB")

    # Read the evaluation labels so each image can be checked against the ground truth.
    print("\n" + "=" * 70)
    print("READING EVALUATION LABELS")
    print("=" * 70)
    df = pd.read_csv(LABEL_FILE)
    print(df.head())
    print("\nColumns:")
    print(df.columns.tolist())

    FILENAME_COLUMN = "filename"
    LABEL_COLUMN = "true_label"

    # Create a dataset that maps each CSV row to an image file and its true label.
    class EvalDataset(Dataset):
        def __init__(self, dataframe, image_dir, transform=None):
            self.dataframe = dataframe.reset_index(drop=True)
            self.image_dir = image_dir
            self.transform = transform

        def __len__(self):
            return len(self.dataframe)

        def __getitem__(self, index):
            row = self.dataframe.iloc[index]
            filename = str(row[FILENAME_COLUMN])
            label = str(row[LABEL_COLUMN])
            image_path = os.path.join(self.image_dir, filename)

            if not os.path.exists(image_path):
                image_path = os.path.join(self.image_dir, os.path.basename(filename))

            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")

            image = Image.open(image_path).convert("RGB")
            if self.transform is not None:
                image = self.transform(image)

            return image, label, filename

    eval_dataset = EvalDataset(dataframe=df, image_dir=EVAL_DIR, transform=transform)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    print("\nNumber of evaluation images:")
    print(len(eval_dataset))

    # Run the model over all validation images and collect scores for each prediction.
    print("\n" + "=" * 70)
    print("RUNNING EVALUATION")
    print("=" * 70)

    all_true = []
    all_pred = []
    all_probabilities = []
    all_filenames = []
    total_inference_time = 0.0

    with torch.no_grad():
        for images, labels, filenames in eval_loader:
            images = images.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start_time = time.perf_counter()

            outputs = model(images)

            if device.type == "cuda":
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            total_inference_time += end_time - start_time

            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(probabilities, dim=1)

            probabilities = probabilities.cpu().numpy()
            predictions = predictions.cpu().numpy()

            all_true.extend(labels)
            all_pred.extend(classes[pred] for pred in predictions)
            all_probabilities.extend(probabilities)
            all_filenames.extend(filenames)

    accuracy = accuracy_score(all_true, all_pred)
    precision_macro = precision_score(all_true, all_pred, labels=classes, average="macro", zero_division=0)
    recall_macro = recall_score(all_true, all_pred, labels=classes, average="macro", zero_division=0)
    f1_macro = f1_score(all_true, all_pred, labels=classes, average="macro", zero_division=0)
    precision_weighted = precision_score(all_true, all_pred, labels=classes, average="weighted", zero_division=0)
    recall_weighted = recall_score(all_true, all_pred, labels=classes, average="weighted", zero_division=0)
    f1_weighted = f1_score(all_true, all_pred, labels=classes, average="weighted", zero_division=0)

    # Compute the main classification metrics from the model predictions.
    print("\n" + "=" * 70)
    print("MODEL PERFORMANCE")
    print("=" * 70)
    print(f"Accuracy           : {accuracy * 100:.2f}%")
    print(f"Macro Precision    : {precision_macro * 100:.2f}%")
    print(f"Macro Recall       : {recall_macro * 100:.2f}%")
    print(f"Macro F1           : {f1_macro * 100:.2f}%")
    print(f"Weighted Precision : {precision_weighted * 100:.2f}%")
    print(f"Weighted Recall    : {recall_weighted * 100:.2f}%")
    print(f"Weighted F1        : {f1_weighted * 100:.2f}%")

    # Write the per-class report and confusion matrix for review.
    print("\n" + "=" * 70)
    print("PER-CLASS PERFORMANCE")
    print("=" * 70)
    report = classification_report(
        all_true,
        all_pred,
        labels=classes,
        target_names=classes,
        digits=4,
        zero_division=0,
    )
    print(report)
    with open(os.path.join(RESULTS_DIR, "classification_report.txt"), "w") as f:
        f.write(report)

    cm = confusion_matrix(all_true, all_pred, labels=classes)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=classes, yticklabels=classes, cmap="Blues")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Basic CNN - Confusion Matrix")
    plt.tight_layout()
    cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()

    predictions_df = pd.DataFrame(
        {
            "filename": all_filenames,
            "true_label": all_true,
            "predicted_label": all_pred,
        }
    )
    max_probabilities = [max(prob) for prob in all_probabilities]
    predictions_df["confidence"] = max_probabilities
    prediction_path = os.path.join(RESULTS_DIR, "predictions.csv")
    predictions_df.to_csv(prediction_path, index=False)

    misclassified = predictions_df[predictions_df["true_label"] != predictions_df["predicted_label"]]
    misclassified_path = os.path.join(RESULTS_DIR, "misclassified.csv")
    misclassified.to_csv(misclassified_path, index=False)

    num_images = len(eval_dataset)
    average_time_ms = (total_inference_time / num_images) * 1000
    images_per_second = num_images / total_inference_time if total_inference_time > 0 else 0

    # Measure average latency and throughput across the full evaluation set.
    print("\n" + "=" * 70)
    print("INFERENCE PERFORMANCE")
    print("=" * 70)
    print(f"Images evaluated       : {num_images}")
    print(f"Total inference time   : {total_inference_time:.4f} sec")
    print(f"Average inference time : {average_time_ms:.3f} ms/image")
    print(f"Throughput             : {images_per_second:.2f} images/sec")

    summary = {
        "model": "BasicCNN",
        "num_classes": num_classes,
        "num_evaluation_images": num_images,
        "total_parameters": total_parameters,
        "model_size_mb": model_size_mb,
        "accuracy": accuracy,
        "macro_precision": precision_macro,
        "macro_recall": recall_macro,
        "macro_f1": f1_macro,
        "weighted_precision": precision_weighted,
        "weighted_recall": recall_weighted,
        "weighted_f1": f1_weighted,
        "average_inference_ms": average_time_ms,
        "images_per_second": images_per_second,
    }
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(os.path.join(RESULTS_DIR, "model_performance.csv"), index=False)

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print("\nGenerated files:")
    print("1. results/model_performance.csv")
    print("2. results/classification_report.txt")
    print("3. results/confusion_matrix.png")
    print("4. results/predictions.csv")
    print("5. results/misclassified.csv")


if __name__ == "__main__":
    main()
