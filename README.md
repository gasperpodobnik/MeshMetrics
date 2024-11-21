# MeshMetrics
> Official Python-based implementation of `MeshMetrics` from [_Metrics Revolutions: Groundbreaking Insights into the Implementation of Metrics for Biomedical Image Segmentation_](https://arxiv.org/abs/2410.02630).

## About
`MeshMetrics` provides a precise, mesh-based implementation of critical metrics used in the evaluation of image segmentation tasks. Quantitative performance metrics are fundamental for objective and reproducible segmentation assessments. Although *overlap-based* metrics - such as **Dice similarity coefficient** (DSC) and **intersection over union** (IoU) - are relatively straightforward to compute, *distance-based* metrics often lack uniform implementation across tools due to the complexity of distance calculations.

`MeshMetrics` includes accurate implementations of key distance-based metrics:
- **Hausdorff Distance** (HD) with $p$-th **percentile variants** (HD<sub>p</sub>)
- **Mean Average Surface Distance** (MASD)
- **Average Symmetric Surface Distance** (ASSD)
- **Normalized Surface Distance** (NSD)
- **Boundary Intersection over Union** (BIoU)

By leveraging mesh representations of segmentation masks, `MeshMetrics` ensures precision in distance and boundary element size calculations. For further details and comparisons with other open-source tools supporting distance-based metric calculations, please refer to [our paper](https://arxiv.org/abs/2410.02630).

If you use `MeshMetrics` in your work, please cite:
```
Podobnik, G., & Vrtovec, T. (2024). Metrics Revolutions: Groundbreaking Insights into the Implementation of Metrics for Biomedical Image Segmentation. arXiv preprint arXiv:2410.02630.
```

## Installation
### System Dependencies
This package requires `libxrender1` to be installed on your system. Install it via:
```bash
sudo apt update && sudo apt install -y libxrender1
```

### Install `MeshMetrics` package
First, clone the repository and install the required dependencies along with the `MeshMetrics` package using pip:
```bash
$ git clone https://github.com/gasperpodobnik/MeshMetrics.git
$ pip install MeshMetrics/
```

## Usage
Simple usage example of `MeshMetrics` for 3D segmentation masks is shown below. See more examples in the [`examples.ipynb`](examples.ipynb) notebook.

```python
from pathlib import Path
from MeshMetrics import DistanceMetrics

data_dir = Path("data")
# initialize DistanceMetrics object
dist_metrics = DistanceMetrics()

# read binary segmentation masks
ref_mask_sitk = sitk.ReadImage(data_dir / "example_3d_ref_mask.nii.gz")
pred_mask_sitk = sitk.ReadImage(data_dir / "example_3d_pred_mask.nii.gz")

# set input masks and spacing (only needed if both inputs are numpy arrays or vtk meshes)
dist_metrics.set_input(ref=ref_mask_sitk, pred=pred_mask_sitk)

# Hausdorff Distance (HD), by default, HD percentile is set to 100 (equivalent to HD)
hd100 = dist_metrics.hd()
# 95th percentile HD
hd95 = dist_metrics.hd(percentile=95)
# Mean Average Surface Distance (MASD)
masd = dist_metrics.masd()
# Average Symmetric Surface Distance (ASSD)
assd = dist_metrics.assd()
# Normalized Surface Distance (NSD) with tau=2
nsd2 = dist_metrics.nsd(tau=2)
# Boundary Intersection over Union (BIoU) with tau=2
biou2 = dist_metrics.biou(tau=2)

# ----------------------------------------
# if loading masks from files with SimpleITK library, note that spacing needs to be reordered
import SimpleITK as sitk
ref_path, pred_path = ..., ...
ref_sitk, pred_sitk = sitk.ReadImage(ref_path), sitk.ReadImage(pred_path)
ref_mask = sitk.GetArrayFromImage(ref_sitk).astype(bool)
pred_mask = sitk.GetArrayFromImage(pred_sitk).astype(bool)

assert ref_sitk.GetSize() == pred_sitk.GetSize()
assert ref_sitk.GetSpacing() == pred_sitk.GetSpacing()
assert ref_sitk.GetOrigin() == pred_sitk.GetOrigin()
assert ref_sitk.GetDirection() == pred_sitk.GetDirection()

# spacing should resemble the order of numpy array axes
spacing = ref_sitk.GetSpacing()[::-1]
# ... follow the same procedure as before
```
