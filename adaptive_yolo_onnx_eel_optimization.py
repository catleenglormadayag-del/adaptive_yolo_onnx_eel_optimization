import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO


# ============================================================
# 1. SETTING THE LOCAL PATHS
# ============================================================

BASE_ROOT = r"C:\EelOnnx"

# Exported YOLO-ONNX model
MODEL_PATH = os.path.join(BASE_ROOT, "best.onnx")

# Dataset used ONLY for threshold optimization
# This should contain your 333 images arranged in folders 1 to 20
OPTIMIZATION_ROOT = os.path.join(BASE_ROOT, "threshold optimization 333")

# Dataset used ONLY for final independent counting evaluation
# This should contain your 1,400 newly captured images arranged in folders 1 to 20
INDEPENDENT_TEST_ROOT = os.path.join(BASE_ROOT, "independent images")

# Output folders
OUTPUT_DIR = os.path.join(BASE_ROOT, "eel_onnx_independent_test_results")
BASELINE_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "baseline_outputs")
OPTIMIZED_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "optimized_outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(BASELINE_OUTPUT_DIR, exist_ok=True)
os.makedirs(OPTIMIZED_OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. LOADING THE YOLO-ONNX MODEL
# ============================================================

print("Loading YOLO-ONNX model...")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

model = YOLO(MODEL_PATH, task="detect")

print("Model loaded successfully.")


# ============================================================
# 3. READING THE IMAGES FROM COUNT FOLDERS
# Folder name = manual eel count
# Example: folder "5" means each image inside has 5 eels
# ============================================================

valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def read_images_from_count_folders(root_path):
    """
    Reads images from folders named according to manual eel count.
    Example:
        root_path/1/image001.jpg means manual count = 1
        root_path/20/image100.jpg means manual count = 20
    """

    if not os.path.exists(root_path):
        raise FileNotFoundError(f"Dataset folder not found: {root_path}")

    image_records = []

    for folder_name in sorted(os.listdir(root_path)):
        folder_path = os.path.join(root_path, folder_name)

        if not os.path.isdir(folder_path):
            continue

        if not folder_name.isdigit():
            continue

        manual_count = int(folder_name)

        for file_name in sorted(os.listdir(folder_path)):
            if file_name.lower().endswith(valid_extensions):
                image_path = os.path.join(folder_path, file_name)

                image_records.append({
                    "image_path": image_path,
                    "image_name": file_name,
                    "folder_name": folder_name,
                    "manual_count": manual_count
                })

    if len(image_records) == 0:
        raise Exception(f"No images found in {root_path}. Check your folder path and count folders.")

    return image_records


optimization_records = read_images_from_count_folders(OPTIMIZATION_ROOT)
independent_test_records = read_images_from_count_folders(INDEPENDENT_TEST_ROOT)

print(f"Threshold optimization images found: {len(optimization_records)}")
print(f"Independent counting test images found: {len(independent_test_records)}")


# ============================================================
# 4. METRIC FUNCTIONS
# ============================================================

def counting_accuracy(manual_count, predicted_count):
    """
    Accuracy = (1 - |manual - predicted| / manual) x 100
    """

    if manual_count <= 0:
        return 0

    error = abs(manual_count - predicted_count)
    accuracy = (1 - error / manual_count) * 100

    return max(0, accuracy)


def compute_mae(actual, predicted):
    actual = np.array(actual, dtype=float)
    predicted = np.array(predicted, dtype=float)
    return np.mean(np.abs(actual - predicted))


def compute_rmse(actual, predicted):
    actual = np.array(actual, dtype=float)
    predicted = np.array(predicted, dtype=float)
    return np.sqrt(np.mean((actual - predicted) ** 2))


def run_count(image_path, conf, iou):
    """
    Run YOLO-ONNX using GPU and return predicted count, inference time, and result object.
    """

    results = model(
        image_path,
        conf=conf,
        iou=iou,
        imgsz=640,
        device=0,
        verbose=False
    )

    predicted_count = len(results[0].boxes)
    inference_time = results[0].speed.get("inference", 0)

    return predicted_count, inference_time, results


# ============================================================
# 5. ADAPTIVE PARAMETER OPTIMIZATION
# This section uses ONLY the 333-image threshold optimization dataset.
# It identifies the best confidence and IoU combination.
# ============================================================

confidence_values = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
iou_values = [0.30, 0.40, 0.45, 0.50, 0.60, 0.70]

threshold_rows = []

print("\nStarting adaptive confidence and IoU optimization...")
print("Dataset used: threshold optimization dataset only")

for conf in confidence_values:
    for iou in iou_values:

        actual_counts = []
        predicted_counts = []
        accuracies = []
        differences = []
        inference_times = []

        print(f"Testing confidence={conf}, IoU={iou}")

        for record in optimization_records:
            image_path = record["image_path"]
            manual_count = record["manual_count"]

            predicted_count, inference_time, _ = run_count(
                image_path=image_path,
                conf=conf,
                iou=iou
            )

            difference = abs(manual_count - predicted_count)
            accuracy = counting_accuracy(manual_count, predicted_count)

            actual_counts.append(manual_count)
            predicted_counts.append(predicted_count)
            accuracies.append(accuracy)
            differences.append(difference)
            inference_times.append(inference_time)

        mae = compute_mae(actual_counts, predicted_counts)
        rmse = compute_rmse(actual_counts, predicted_counts)

        threshold_rows.append({
            "Confidence": conf,
            "IoU": iou,
            "No. of Optimization Images": len(optimization_records),
            "Minimum Manual Count": int(min(actual_counts)),
            "Maximum Manual Count": int(max(actual_counts)),
            "Mean Accuracy (%)": round(np.mean(accuracies), 2),
            "SD Accuracy": round(np.std(accuracies), 2),
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "Mean Absolute Difference": round(np.mean(differences), 4),
            "Mean Inference Time (ms)": round(np.mean(inference_times), 2)
        })

df_thresholds = pd.DataFrame(threshold_rows)

df_thresholds_ranked = df_thresholds.sort_values(
    by=["Mean Accuracy (%)", "MAE", "RMSE", "Mean Inference Time (ms)"],
    ascending=[False, True, True, True]
).reset_index(drop=True)

df_best = df_thresholds_ranked.head(1)

BEST_CONF = float(df_best.iloc[0]["Confidence"])
BEST_IOU = float(df_best.iloc[0]["IoU"])

print("\nBest optimized configuration identified using threshold optimization dataset:")
print(f"Best Confidence: {BEST_CONF}")
print(f"Best IoU: {BEST_IOU}")
print(df_best)


# ============================================================
# 6. BASELINE VS OPTIMIZED COMPARISON
# This section uses ONLY the 1,400 newly captured independent test images.
# These images were NOT used in training and NOT used in threshold optimization.
# ============================================================

BASELINE_CONF = 0.25
BASELINE_IOU = 0.70

OPTIMIZED_CONF = BEST_CONF
OPTIMIZED_IOU = BEST_IOU

comparison_rows = []

print("\nRunning baseline vs optimized comparison...")
print("Dataset used: independent counting test dataset only")

for idx, record in enumerate(independent_test_records, start=1):
    image_path = record["image_path"]
    image_name = record["image_name"]
    manual_count = record["manual_count"]
    folder_name = record["folder_name"]

    safe_name = f"manual{manual_count}_{os.path.splitext(image_name)[0]}.jpg"

    baseline_output_path = os.path.join(BASELINE_OUTPUT_DIR, f"baseline_{safe_name}")
    optimized_output_path = os.path.join(OPTIMIZED_OUTPUT_DIR, f"optimized_{safe_name}")

    # Baseline YOLO-ONNX
    baseline_count, baseline_time, baseline_results = run_count(
        image_path=image_path,
        conf=BASELINE_CONF,
        iou=BASELINE_IOU
    )

    baseline_results[0].save(filename=baseline_output_path)

    baseline_difference = abs(manual_count - baseline_count)
    baseline_accuracy = counting_accuracy(manual_count, baseline_count)

    # Optimized YOLO-ONNX
    optimized_count, optimized_time, optimized_results = run_count(
        image_path=image_path,
        conf=OPTIMIZED_CONF,
        iou=OPTIMIZED_IOU
    )

    optimized_results[0].save(filename=optimized_output_path)

    optimized_difference = abs(manual_count - optimized_count)
    optimized_accuracy = counting_accuracy(manual_count, optimized_count)

    comparison_rows.append({
        "No.": idx,
        "Folder Manual Count": folder_name,
        "Image": image_name,
        "Image Path": image_path,

        "Manual Count": manual_count,

        "Baseline Confidence": BASELINE_CONF,
        "Baseline IoU": BASELINE_IOU,
        "Baseline Count": baseline_count,
        "Baseline Difference": baseline_difference,
        "Baseline Accuracy (%)": round(baseline_accuracy, 2),
        "Baseline Inference Time (ms)": round(baseline_time, 2),

        "Optimized Confidence": OPTIMIZED_CONF,
        "Optimized IoU": OPTIMIZED_IOU,
        "Optimized Count": optimized_count,
        "Optimized Difference": optimized_difference,
        "Optimized Accuracy (%)": round(optimized_accuracy, 2),
        "Optimized Inference Time (ms)": round(optimized_time, 2),

        "Accuracy Improvement (%)": round(optimized_accuracy - baseline_accuracy, 2),
        "Error Reduction": baseline_difference - optimized_difference
    })

    print(f"Processed {idx}/{len(independent_test_records)}: {image_name}")


df_comparison = pd.DataFrame(comparison_rows)


# ============================================================
# 7. SUMMARY TABLE OF FINAL INDEPENDENT TEST RESULT
# ============================================================

actual_counts = df_comparison["Manual Count"].astype(float)

baseline_predictions = df_comparison["Baseline Count"].astype(float)
optimized_predictions = df_comparison["Optimized Count"].astype(float)

summary_rows = [
    {
        "Method": "Baseline YOLO-ONNX",
        "Confidence": BASELINE_CONF,
        "IoU": BASELINE_IOU,
        "No. of Independent Test Images": len(df_comparison),
        "Minimum Manual Count": int(df_comparison["Manual Count"].min()),
        "Maximum Manual Count": int(df_comparison["Manual Count"].max()),
        "Mean Accuracy (%)": round(df_comparison["Baseline Accuracy (%)"].mean(), 2),
        "SD Accuracy": round(df_comparison["Baseline Accuracy (%)"].std(), 2),
        "MAE": round(compute_mae(actual_counts, baseline_predictions), 4),
        "RMSE": round(compute_rmse(actual_counts, baseline_predictions), 4),
        "Mean Inference Time (ms)": round(df_comparison["Baseline Inference Time (ms)"].mean(), 2)
    },
    {
        "Method": "Optimized YOLO-ONNX",
        "Confidence": OPTIMIZED_CONF,
        "IoU": OPTIMIZED_IOU,
        "No. of Independent Test Images": len(df_comparison),
        "Minimum Manual Count": int(df_comparison["Manual Count"].min()),
        "Maximum Manual Count": int(df_comparison["Manual Count"].max()),
        "Mean Accuracy (%)": round(df_comparison["Optimized Accuracy (%)"].mean(), 2),
        "SD Accuracy": round(df_comparison["Optimized Accuracy (%)"].std(), 2),
        "MAE": round(compute_mae(actual_counts, optimized_predictions), 4),
        "RMSE": round(compute_rmse(actual_counts, optimized_predictions), 4),
        "Mean Inference Time (ms)": round(df_comparison["Optimized Inference Time (ms)"].mean(), 2)
    }
]

df_summary = pd.DataFrame(summary_rows)


# ============================================================
# 8. RESULT BY MANUAL COUNT FOLDER
# ============================================================

df_by_manual_count = df_comparison.groupby("Manual Count").agg(
    Test_Images=("Image", "count"),

    Baseline_Mean_Count=("Baseline Count", "mean"),
    Baseline_Mean_Accuracy=("Baseline Accuracy (%)", "mean"),
    Baseline_MAE=("Baseline Difference", "mean"),

    Optimized_Mean_Count=("Optimized Count", "mean"),
    Optimized_Mean_Accuracy=("Optimized Accuracy (%)", "mean"),
    Optimized_MAE=("Optimized Difference", "mean"),

    Mean_Accuracy_Improvement=("Accuracy Improvement (%)", "mean"),
    Mean_Error_Reduction=("Error Reduction", "mean")
).reset_index()

df_by_manual_count = df_by_manual_count.round(2)


# ============================================================
# 9. FINAL IMPROVEMENT SUMMARY
# ============================================================

baseline_summary = df_summary[df_summary["Method"] == "Baseline YOLO-ONNX"].iloc[0]
optimized_summary = df_summary[df_summary["Method"] == "Optimized YOLO-ONNX"].iloc[0]

accuracy_gain = optimized_summary["Mean Accuracy (%)"] - baseline_summary["Mean Accuracy (%)"]
mae_reduction = baseline_summary["MAE"] - optimized_summary["MAE"]
rmse_reduction = baseline_summary["RMSE"] - optimized_summary["RMSE"]

df_final_improvement = pd.DataFrame([
    {
        "Baseline Accuracy (%)": baseline_summary["Mean Accuracy (%)"],
        "Optimized Accuracy (%)": optimized_summary["Mean Accuracy (%)"],
        "Accuracy Gain (percentage points)": round(accuracy_gain, 2),

        "Baseline MAE": baseline_summary["MAE"],
        "Optimized MAE": optimized_summary["MAE"],
        "MAE Reduction": round(mae_reduction, 4),

        "Baseline RMSE": baseline_summary["RMSE"],
        "Optimized RMSE": optimized_summary["RMSE"],
        "RMSE Reduction": round(rmse_reduction, 4),

        "Baseline Mean Inference Time (ms)": baseline_summary["Mean Inference Time (ms)"],
        "Optimized Mean Inference Time (ms)": optimized_summary["Mean Inference Time (ms)"]
    }
])


# ============================================================
# 10. SAVE EXCEL AND CSV RESULTS
# ============================================================

comparison_excel_path = os.path.join(
    OUTPUT_DIR,
    "independent_test_baseline_vs_optimized_yolo_onnx.xlsx"
)

threshold_excel_path = os.path.join(
    OUTPUT_DIR,
    "threshold_optimization_333_results.xlsx"
)

with pd.ExcelWriter(comparison_excel_path, engine="openpyxl") as writer:
    df_summary.to_excel(writer, sheet_name="Final Summary Independent", index=False)
    df_final_improvement.to_excel(writer, sheet_name="Final Improvement", index=False)
    df_comparison.to_excel(writer, sheet_name="Image Level Independent", index=False)
    df_by_manual_count.to_excel(writer, sheet_name="By Manual Count", index=False)
    df_thresholds_ranked.to_excel(writer, sheet_name="Threshold Optimization 333", index=False)

df_thresholds_ranked.to_excel(threshold_excel_path, index=False)

df_summary.to_csv(
    os.path.join(OUTPUT_DIR, "final_summary_independent_test.csv"),
    index=False
)

df_final_improvement.to_csv(
    os.path.join(OUTPUT_DIR, "final_improvement_independent_test.csv"),
    index=False
)

df_comparison.to_csv(
    os.path.join(OUTPUT_DIR, "image_level_independent_test.csv"),
    index=False
)

df_by_manual_count.to_csv(
    os.path.join(OUTPUT_DIR, "results_by_manual_count_folder_independent_test.csv"),
    index=False
)

df_thresholds_ranked.to_csv(
    os.path.join(OUTPUT_DIR, "threshold_optimization_333_results.csv"),
    index=False
)

print("\nSaved tables successfully.")


# ============================================================
# 11. FIGURE: FINAL INDEPENDENT TEST ACCURACY COMPARISON
# ============================================================

plt.figure(figsize=(10, 6))

methods = df_summary["Method"]
accuracy_values = df_summary["Mean Accuracy (%)"]

plt.bar(methods, accuracy_values)
plt.ylabel("Mean Counting Accuracy (%)")
plt.xlabel("Method")
plt.title("Final Independent Test: Baseline YOLO-ONNX vs Optimized YOLO-ONNX")
plt.ylim(0, 100)

for i, value in enumerate(accuracy_values):
    plt.text(i, value + 1, f"{value:.2f}%", ha="center")

plt.tight_layout()

accuracy_graph_path = os.path.join(
    OUTPUT_DIR,
    "figure_final_independent_test_accuracy.png"
)

plt.savefig(accuracy_graph_path, dpi=300)
plt.close()


# ============================================================
# 12. FIGURE: FINAL INDEPENDENT TEST MAE AND RMSE
# ============================================================

plt.figure(figsize=(10, 6))

x = np.arange(len(df_summary["Method"]))
width = 0.35

mae_values = df_summary["MAE"]
rmse_values = df_summary["RMSE"]

plt.bar(x - width / 2, mae_values, width, label="MAE")
plt.bar(x + width / 2, rmse_values, width, label="RMSE")

plt.xticks(x, df_summary["Method"])
plt.ylabel("Error Value")
plt.xlabel("Method")
plt.title("Final Independent Test: Error Metrics of Baseline and Optimized YOLO-ONNX")
plt.legend()

for i, value in enumerate(mae_values):
    plt.text(i - width / 2, value + 0.02, f"{value:.2f}", ha="center")

for i, value in enumerate(rmse_values):
    plt.text(i + width / 2, value + 0.02, f"{value:.2f}", ha="center")

plt.tight_layout()

error_graph_path = os.path.join(
    OUTPUT_DIR,
    "figure_final_independent_test_error_metrics.png"
)

plt.savefig(error_graph_path, dpi=300)
plt.close()


# ============================================================
# 13. FIGURE: CONFIDENCE-IOU HEATMAP
# This heatmap is from the 333-image threshold optimization dataset.
# Do not label this as final independent test result.
# ============================================================

heatmap_data = df_thresholds.pivot(
    index="Confidence",
    columns="IoU",
    values="Mean Accuracy (%)"
)

plt.figure(figsize=(10, 7))
plt.imshow(heatmap_data, aspect="auto")

plt.xticks(
    ticks=np.arange(len(heatmap_data.columns)),
    labels=heatmap_data.columns
)

plt.yticks(
    ticks=np.arange(len(heatmap_data.index)),
    labels=heatmap_data.index
)

plt.xlabel("IoU Threshold")
plt.ylabel("Confidence Threshold")
plt.title("Threshold Optimization Set: Effect of Confidence and IoU on Mean Counting Accuracy")

cbar = plt.colorbar()
cbar.set_label("Mean Counting Accuracy (%)")

for i in range(len(heatmap_data.index)):
    for j in range(len(heatmap_data.columns)):
        value = heatmap_data.iloc[i, j]
        plt.text(j, i, f"{value:.1f}", ha="center", va="center")

plt.tight_layout()

heatmap_path = os.path.join(
    OUTPUT_DIR,
    "figure_threshold_optimization_confidence_iou_heatmap.png"
)

plt.savefig(heatmap_path, dpi=300)
plt.close()


# ============================================================
# 14. SAVE TEXT SUMMARY 
# ============================================================

text_summary_path = os.path.join(OUTPUT_DIR, "manuscript_result_summary.txt")

with open(text_summary_path, "w", encoding="utf-8") as f:
    f.write("YOLO-ONNX Eel Counting Result Summary\n")
    f.write("=====================================\n\n")

    f.write("Dataset Separation:\n")
    f.write(f"- Threshold optimization dataset: {len(optimization_records)} images\n")
    f.write(f"- Independent counting test dataset: {len(independent_test_records)} images\n\n")

    f.write("Threshold Selection:\n")
    f.write(f"- Selected confidence threshold: {BEST_CONF}\n")
    f.write(f"- Selected IoU threshold: {BEST_IOU}\n")
    f.write("- The threshold was selected using only the threshold optimization dataset.\n\n")

    f.write("Final Independent Test Results:\n")
    f.write(df_summary.to_string(index=False))
    f.write("\n\n")

    f.write("Final Improvement:\n")
    f.write(df_final_improvement.to_string(index=False))
    f.write("\n\n")

    f.write("Important Interpretation:\n")
    f.write(
        "The final accuracy, MAE, and RMSE are based only on the independent "
        "counting test dataset. The independent test images were not used during "
        "YOLOv8n training, validation, testing, or threshold optimization.\n"
    )


# ============================================================
# 15. ZIP ALL RESULTS
# ============================================================

zip_base = os.path.join(BASE_ROOT, "eel_onnx_independent_test_results")
shutil.make_archive(zip_base, "zip", OUTPUT_DIR)

print("\nDONE.")
print("Best Confidence selected from 333-image threshold optimization dataset:", BEST_CONF)
print("Best IoU selected from 333-image threshold optimization dataset:", BEST_IOU)

print("\nFinal independent test summary:")
print(df_summary)

print("\nFinal improvement:")
print(df_final_improvement)

print("\nMain Excel result:")
print(comparison_excel_path)

print("\nOutput folder:")
print(OUTPUT_DIR)

print("\nZipped results:")
print(zip_base + ".zip")
