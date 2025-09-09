# MeshMetrics
> Official Python-based implementation of `MeshMetrics` from [_MeshMetrics: A Precise Implementation of Distance-Based Image Segmentation Metrics_](https://doi.org/10.48550/arXiv.2509.05670), motivated by the implementation pitfalls identified in [_Understanding Implementation Pitfalls of Distance-Based Metrics for Image Segmentation_](https://doi.org/10.48550/arXiv.2410.02630) and [_HDilemma: Are Open-Source Hausdorff Distance Implementations Equivalent?_](https://link.springer.com/chapter/10.1007/978-3-031-72114-4_30)

## About
`MeshMetrics` is a precise, mesh-based implementation of widely used distance-based metrics for evaluating image segmentation tasks. By leveraging mesh representations of segmentation, `MeshMetrics` ensures precision in distance and boundary element size calculations. For a detailed description and a comparison with other open-source tools supporting distance-based metric calculations, see [our paper](https://doi.org/10.48550/arXiv.2509.05670).

The library supports both 2D and 3D data and works seamlessly with multiple segmentation formats (`numpy.ndarray`, `SimpleITK.Image`, and `vtk.vtkPolyData`). It also allows mixing representations between reference and predicted segmentations - for example, one input can be a mask image (`SimpleITK.Image`), while the other is a surface mesh (`vtk.vtkPolyData`). See the *Advanced usage* section in [`examples.ipynb`](examples.ipynb) for more details.

Available distance-based metrics:
- **Hausdorff distance** (HD) with $p$-th **percentile variants** (HD<sub>p</sub>)
- **Mean average surface distance** (MASD)
- **Average symmetric surface distance** (ASSD)
- **Normalized surface distance** (NSD)
- **Boundary intersection over union** (BIoU)

For convenience, `MeshMetrics` also includes implementations of the **Dice similarity coefficient** (DSC) and **intersection over union** (IoU).

![overview](./data/paper_overview.png)

If you use `MeshMetrics` in your work, please cite:
```
Podobnik, G., & Vrtovec, T. (2025). MeshMetrics: A Precise Implementation of Distance-Based Image Segmentation Metrics. arXiv preprint arXiv:2509.05670.
Podobnik, G., & Vrtovec, T. (2025). Understanding Implementation Pitfalls of Distance-Based Metrics for Image Segmentation. arXiv preprint arXiv:2410.02630.
```

## Installation
### System Dependencies
This package requires `libxrender1` to be installed on your system. Install it via:
```bash
sudo apt update && sudo apt install -y libxrender1
```

### Install `MeshMetrics` package
Clone the repository and install the required dependencies along with the `MeshMetrics` package using pip:
```bash
$ pip install git+https://github.com/gasperpodobnik/MeshMetrics.git
```

## Usage
Simple usage example of `MeshMetrics` for 3D segmentation masks is shown below.
See [`examples.ipynb`](examples.ipynb) notebook for more examples.

```python
from pathlib import Path
import SimpleITK as sitk
from MeshMetrics import DistanceMetrics

data_dir = Path("data")
# initialize DistanceMetrics object
dist_metrics = DistanceMetrics()

# read binary segmentation masks
ref_sitk = sitk.ReadImage(str(data_dir / "example_3d_ref_mask.nii.gz"))
pred_sitk = sitk.ReadImage(str(data_dir / "example_3d_pred_mask.nii.gz"))

# Set parameters
percentile = 95  # percentile for HD
tau = 2.0  # tolerance for NSD and BIoU

# Initialize distance metrics class
mesh_metrics = DistanceMetrics()

## ----- example (2D) -----
mesh_metrics.set_input(sitk_mask1, sitk_mask2)
# store flags indicating empty masks
results = {
    "ref_is_empty": mesh_metrics.ref_is_empty,
    "pred_is_empty": mesh_metrics.pred_is_empty,
}
# Hausdorff distance (HD), by default, HD percentile is set to 100 (equivalent to HD)
results["HD_100"] = mesh_metrics.hd()
# p-th percentile HD (HD_p)
results[f"HD_{percentile}"] = mesh_metrics.hd(percentile=percentile)
# Mean average surface distance (MASD)
results["MASD"] = mesh_metrics.masd()
# Average symmetric surface distance (ASSD)
results["ASSD"] = mesh_metrics.assd()
# Normalized surface distance (NSD) with tau
results[f"NSD_{tau}"] = mesh_metrics.nsd(tau=tau)
# Boundary intersection over union (BIoU) with tau
results[f"BIoU_{tau}"] = mesh_metrics.biou(tau=tau)

# print metric values
units = {"HD": "mm", "MASD": "mm", "ASSD": "mm", "NSD": "%", "BIoU": "%"}
for k, v in results.items():
    unit = units.get(k.split("_")[0], "")
    f = 100.0 if unit == "%" else 1.0
    print(f"{k}: {v*f:.2f} {unit}")

# ----------------------------------------
# If using `numpy.ndarray` representations, note that the spacing must be
# reordered when converting a `SimpleITK.Image` object to a `numpy.ndarray`
ref_np = sitk.GetArrayFromImage(ref_sitk).astype(bool)
pred_np = sitk.GetArrayFromImage(pred_sitk).astype(bool)

# spacing should resemble the order of numpy array axes
spacing = ref_np.GetSpacing()[::-1]

dist_metrics = DistanceMetrics()
dist_metrics.set_input(ref=ref_np, pred=pred_np, spacing=spacing)
# ... follow the same procedure as before
```
