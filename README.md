# DeepLearningBasic
Neural Network Implementation and Analysis Assignment

# Kaggle Notebook: Model Optimization and Compression (11 Task ) - `neural-net-assignment.ipynb`

This notebook (`neural-net-assignment.ipynb`) implements various model optimization and compression techniques on a Convolutional Neural Network (CNN) for image classification. It aims to reduce model size and improve inference speed with minimal impact on performance, comparing different strategies: **Pruning**, **Quantization**, and **Knowledge Distillation**.

The code for Task 11 is organized into multiple distinct cells within the notebook, each performing a specific part of the implementation and analysis.

## 1. Dataset

This notebook uses a subset of the **RVL-CDIP** dataset. Specifically, it expects the "RVL-CDIP Small" dataset, which should be available as an input dataset in your Kaggle environment.

**Expected Dataset Path:** `/kaggle/input/rvl-cdip-small/data`

If your dataset is located elsewhere, you will need to update the `data_path` variable in the notebook's data loading cell.

## 2. Kaggle Notebook Setup

To run this notebook on Kaggle, follow these steps:

1.  **Create a New Notebook / Upload Existing:**
    * If you're starting fresh, click on "New Notebook" on Kaggle.
    * If you're uploading your existing `neural-net-assignment.ipynb` file, navigate to "File" > "Upload Notebook" in the Kaggle notebook interface, or upload it directly to your Kaggle workspace.
2.  **Add Data:**
    * In the right-hand sidebar, click on "Add Data".
    * Search for "rvl-cdip small" or similar.
    * Select the dataset (ensure it's the one that extracts to `data` subfolder).
    * Click "Add" to attach it to your notebook.
3.  **GPU Accelerator (Recommended):**
    * In the right-hand sidebar, go to "Settings" -> "Accelerator".
    * Change "None" to "GPU P100" or "GPU T4 x2" for faster training. PyTorch models benefit significantly from GPU acceleration.
4.  **Internet Connection (Optional but Recommended):**
    * Also in "Settings", ensure "Internet" is set to "On" if you need to download any external libraries or models not pre-installed in the Kaggle environment (though for this task, standard libraries should suffice).

## 3. Running the Notebook

The notebook (`neural-net-assignment.ipynb`) is designed to be run sequentially, cell by cell. Each cell performs a distinct part of the overall task.

**Steps to run:**

1.  **Open the Notebook:** Open `neural-net-assignment.ipynb` in the Kaggle environment.
2.  **Run All Cells:** The easiest way to execute the entire notebook is to use the "Run All" option.
    * Go to `Run` menu -> `Run All Cells`.
    * Alternatively, you can click on each cell and press `Shift + Enter` to execute it sequentially.

**Breakdown of Expected Cells (Based on typical structure of Task 11):**

* **Cell 1: Imports and Device Configuration:** Initializes necessary libraries (torch, torchvision, matplotlib, etc.) and sets up the device (CPU/GPU).
* **Cell 2: Data Loading and Preprocessing:** Defines transformations, loads the RVL-CDIP subset, and creates `DataLoader`s for training, validation, and testing.
* **Cell 3: CNN Architectures:** Defines the `SimpleCNN` (Task 7 base model/teacher) and `SmallCNN` (student model for Knowledge Distillation).
* **Cell 4: Helper Functions:** Contains utility functions for `get_model_size`, `count_trainable_parameters`, `measure_inference_time`, and the generalized `train_model` function (which supports KD).
* **Cell 5: Baseline CNN Training:** Trains the `SimpleCNN` for 20 epochs to establish a performance baseline and create the teacher model.
* **Cell 6: Pruning Implementation (10%):** Prunes the baseline model by 10% and fine-tunes it for 10 epochs.
* **Cell 7: Pruning Implementation (30%):** Prunes the baseline model by 30% and fine-tunes it for 10 epochs.
* **Cell 8: Pruning Implementation (50%):** Prunes the baseline model by 50% and fine-tunes it for 10 epochs.
* **Cell 9: Pruning Implementation (70%):** Prunes the baseline model by 70% and fine-tunes it for 10 epochs.
* **Cell 10: Quantization Implementation:** Applies post-training static 8-bit quantization to the baseline model.
* **Cell 11: Knowledge Distillation Implementation:** Trains the `SmallCNN` using the previously trained `SimpleCNN` as a teacher.
* **Cell 12: Analysis and Plotting:** Generates the comparison plots (validation accuracy before/after compression) and the final summary table of test accuracy, model size, and inference time for all models.

**Important Note:** The training processes for the baseline, pruned models, and the student model can take some time, especially on CPU. Using a GPU accelerator is highly recommended for faster execution.

## 4. Expected Output

After successfully running all cells, you should see:

* **Console Output:**
    * Training logs for the baseline CNN (20 epochs).
    * Training logs for each pruned CNN (10 epochs each).
    * Messages for quantization calibration and final metrics.
    * Training logs for the Knowledge Distillation student CNN (10 epochs).
    * A final summary table comparing:
        * Test Accuracy
        * Model Size (MB)
        * Inference Time (ms/image)
        * Trainable Parameters
        for the Baseline CNN, each Pruned CNN, Quantized CNN, and the Student CNN (KD).
* **Plots:** A single bar plot showing the validation accuracy of the Baseline CNN and each of the compressed models.

This output will allow you to analyze the trade-offs between model size, inference speed, and performance for each compression technique.
