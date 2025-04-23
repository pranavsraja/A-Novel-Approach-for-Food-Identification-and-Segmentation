# A-Novel-Approach-for-Food-Identification-and-Segmentation

This repository presents a semantic segmentation pipeline trained on the **FoodSeg103** dataset using **DeepLabV3+ with a ResNet-50 backbone**. 
---

## Dataset

- **Dataset**: [FoodSeg103](https://github.com/Hong-Xiang/FoodSeg103)
- Contains **104 food categories**
- Image and mask pairs are organized into:
  ```
  FoodSeg103/
  ├── Images/
  │   └── img_dir/{train, test}
  │   └── ann_dir/{train, test}
  ├── ImageSets/
  │   └── train.txt
  │   └── test.txt
  ├── category_id.txt
  ```

---

## Project Features

### Data Handling

- Custom `FoodSegDataset` class handles image-mask loading with `.txt` split files
- Robust error handling and mask pre-checks
- Dynamic resizing and interpolation modes based on type

### Augmentation & Preprocessing

- `train` transformations: Random crop, horizontal flip, affine, color jitter, blur
- `val/test` transformations: Center crop + normalization
- All resized to `512x512`

### Class Balancing

- Class-wise frequencies calculated by scanning downsampled masks
- **WeightedRandomSampler** improves sampling for rare classes

---

## Model Architecture

- Base: **DeepLabV3+ with pretrained ResNet-50**
- Modified final classifier layer to output `104` classes
- Uses `CrossEntropyLoss` for multi-class segmentation

---

## Training Strategy

- Optimizer: `AdamW` with `weight_decay=1e-5`
- Scheduler: `ReduceLROnPlateau`
- AMP (Automatic Mixed Precision) via `torch.cuda.amp`
- Early stopping and checkpoint saving for best validation IoU

---

## Evaluation

- Evaluated using:
  - **Pixel Accuracy**
  - **IoU**
  - **Class-wise frequency mapping**
- All metrics and visual outputs are stored.

---

## Output Visuals

Each image includes:
- Original image
- Predicted mask (colored by class ID)
- Overlay image with mask superimposed

---

## How to Run (Colab)

1. Mount Google Drive and unzip dataset
2. Set paths:
```python
base_dir = "/content/drive/MyDrive/Task 1 Food Segmentation Dataset/FoodSeg103/Images"
```
3. Execute cells in order for training, evaluation, and visualization

---

## Libraries Used

- PyTorch  
- Torchvision  
- NumPy, Pandas  
- PIL, OpenCV  
- Matplotlib  
- tqdm, os, glob

---

## Future Improvements

- Use HRNet or SwinTransformer as backbone
- Integrate Dice loss or Focal loss
- Hyperparameter search with Optuna
- Evaluate on custom food datasets (cross-dataset generalization)

---

## Author

- **Pranav Sunil Raja**  
- GitHub: [@pranavsraja](https://github.com/pranavsraja)  
- MSc Data Science & AI, Newcastle University

---

## License

For academic and education use only. Dataset used under the FoodSeg103 terms.
