# Adaptive YOLO-ONNX Eel Counting Optimization

This repository contains the source code for the adaptive threshold optimization of a YOLO-ONNX-based eel counting model. The pipeline was developed to improve eel counting accuracy by selecting the best confidence and Intersection over Union (IoU) thresholds for object detection-based eel counting.

The system compares the performance of a baseline YOLO-ONNX inference configuration with a code-selected optimized configuration using counting accuracy, mean absolute error (MAE), root mean square error (RMSE), standard deviation of accuracy, and inference time.

## Overview
Counting eels using object detection is challenging because eels have elongated bodies and often overlap with one another. These conditions may cause missed detections, false detections, duplicate bounding boxes, and counting errors. To address this problem, this pipeline performs adaptive threshold optimization by testing several confidence and IoU threshold combinations.

This pipeline was also developed to help developers and researchers select a suitable confidence and IoU threshold based on their own dataset and counting performance. Instead of relying only on the default YOLO threshold values or adopting threshold settings from related literature, the system uses actual dataset results to identify the most effective configuration. This makes the threshold selection more data-driven and more appropriate for the specific object, environment, and counting condition being tested.

The best threshold configuration is selected based on the highest mean counting accuracy, lower MAE, lower RMSE, and lower inference time. The optimized YOLO-ONNX output is then compared with the baseline YOLO-ONNX setting to determine whether the optimized thresholds improve counting performance.

## Main Features

* Loads a YOLO-ONNX eel detection model.
* Reads test images from folder-based manual count groups.
* Uses the folder name as the manual eel count or ground truth count.
* Tests multiple confidence and IoU threshold combinations.
* Automatically selects the best threshold configuration.
* Compares baseline YOLO-ONNX and optimized YOLO-ONNX counting performance.
* Computes counting accuracy, MAE, RMSE, standard deviation of accuracy, and inference time.
* Saves image-level and summary results in Excel and CSV formats.
* Saves baseline and optimized detection output images with bounding boxes.
* Generates graphs for accuracy comparison, error metrics, and confidence-IoU heatmap.
* Compresses all generated outputs into a ZIP file.

## Repository File

The main script is:

```text
adaptive_yolo_onnx_eel_optimization.py
```

## Dataset Folder Structure

The test images must be organized in folders where each folder name represents the manual eel count.

C:\EelOnnx
│
├── best.onnx
├── adaptive_yolo_onnx_eel_optimization.py
├── optimized_yolo_onnx_eel_testing.py
│
├── threshold optimization 333
│   │
│   ├── 1
│   │   ├── image001.jpg
│   │   ├── image002.jpg
│   │
│   ├── 2
│   │   ├── image003.jpg
│   │   ├── image004.jpg
│   │
│   ├── 3
│   │   ├── image005.jpg
│   │
│   ├── ...
│   │
│   └── 20
│       ├── image333.jpg
│
├── independent images
│   │
│   ├── 1
│   │   ├── test_image001.jpg
│   │   ├── test_image002.jpg
│   │
│   ├── 2
│   │   ├── test_image003.jpg
│   │   ├── test_image004.jpg
│   │
│   ├── 3
│   │   ├── test_image005.jpg
│   │
│   ├── ...
│   │
│   └── 20
│       ├── test_image333.jpg

Images inside folder `1` have a manual count of 1 eel, images inside folder `2` have a manual count of 2 eels, and so on.

## Model File

The ONNX model must be placed in the working directory. By default, the script uses:

```text
C:\EelOnnx\best.onnx
```

The model path can be changed in the script:

```python
MODEL_PATH = r"C:\EelOnnx\best.onnx"
```

## Requirements

Install the required Python packages before running the script:

```bash
pip install ultralytics pandas numpy matplotlib openpyxl
```

The main libraries used are:

* Ultralytics YOLO
* Pandas
* NumPy
* Matplotlib
* OpenPyXL

## How to Run

Run the script using:

```bash
python adaptive_yolo_onnx_eel_optimization.py
```

Before running the script, make sure that:

1. The ONNX model exists in the specified path.
2. The test images are arranged in numeric folders based on manual eel count.
3. The folder names are numeric values, such as `1`, `2`, `3`, up to `20`.
4. The required Python packages are installed.

## Baseline Configuration

The baseline YOLO-ONNX inference configuration used in the script is:

```text
Confidence = 0.25
IoU = 0.70
```

This baseline configuration represents the initial inference setting before adaptive threshold optimization.

## Adaptive Threshold Optimization

The script evaluates the following confidence threshold values:

```text
0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50
```

The script evaluates the following IoU threshold values:

```text
0.30, 0.40, 0.45, 0.50, 0.60, 0.70
```

The best configuration is selected by ranking the results based on:

1. Highest mean counting accuracy
2. Lowest MAE
3. Lowest RMSE
4. Lowest mean inference time

## Evaluation Metrics

The predicted eel count is computed as the number of detected bounding boxes:

```text
C_predicted = N_boxes
```

Counting accuracy is computed as:

```text
Accuracy = (1 − |C_manual − C_predicted| / C_manual) × 100
```

Mean absolute error is computed as:

```text
MAE = (1/N) Σ |C_manual − C_predicted|
```

Root mean square error is computed as:

```text
RMSE = [(1/N) Σ (C_manual − C_predicted)^2]^1/2
```

where:

* `C_manual` is the manual or ground truth eel count.
* `C_predicted` is the predicted eel count generated by the model.
* `N_boxes` is the total number of detected bounding boxes.
* `N` is the total number of test images.

## Output Folder

The generated results are saved in:

```text
C:\EelOnnx\eel_onnx_optimization_results
```

## Generated Excel Files

The script generates the following Excel files:

```text
baseline_vs_code_selected_optimized_yolo_onnx.xlsx
adaptive_confidence_iou_optimization_results.xlsx
```

The main Excel file contains the following sheets:

```text
Summary Results
Image Level Results
By Manual Count
Threshold Optimization
```

## Generated CSV Files

The script generates the following CSV files:

```text
summary_baseline_vs_optimized.csv
image_level_baseline_vs_optimized.csv
results_by_manual_count_folder.csv
threshold_optimization_results.csv
```

## Generated Figures

The script generates the following figures:

```text
figure3_baseline_vs_optimized_accuracy.png
figure4_baseline_vs_optimized_error_metrics.png
figure6_confidence_iou_accuracy_heatmap.png
```

These figures show:

* Baseline versus optimized mean counting accuracy
* MAE and RMSE comparison
* Effect of confidence and IoU thresholds on mean counting accuracy

## Detection Output Images

The script saves detection output images in two folders:

```text
baseline_outputs
optimized_outputs
```

The `baseline_outputs` folder contains images processed using the baseline YOLO-ONNX setting.

The `optimized_outputs` folder contains images processed using the optimized threshold setting selected by the script.

## Zipped Results

After all outputs are generated, the script compresses the result folder into:

```text
eel_onnx_optimization_results.zip
```

## Main Result Summary

Using the eel counting image dataset, the baseline YOLO-ONNX model obtained a mean counting accuracy of 83.08%. After adaptive threshold optimization, the optimized YOLO-ONNX model obtained a mean counting accuracy of 94.93%. This represents an improvement of 11.85 percentage points.

The optimized model also reduced the counting error based on MAE and RMSE.

## Notes

* The confidence and IoU thresholds are applied during ONNX inference and counting evaluation.
* These thresholds are not training parameters.
* The baseline setting represents the initial inference configuration before threshold optimization.
* The optimized setting is selected by the script based on counting accuracy and error metrics.
* The folder name is used as the manual eel count for each image inside that folder.
* The script is intended for eel counting images where overlapping and occlusion may affect detection performance.

## Source Code Availability

The source code and configuration files for the proposed YOLO-ONNX eel counting optimization pipeline are publicly available at:

```text
https://github.com/catleenglormadayag-del/adaptive_yolo_onnx_eel_optimization
```

## Author

Catleen Glo M. Feliciano
Isabela State University
