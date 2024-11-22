from functools import lru_cache
import logging
from typing import Tuple, Union

import numpy as np
import vtk
import SimpleITK as sitk
from .utils import (
    vtk_meshing,
    np2sitk,
    sitk2np,
    vtk_centroids2contour_measurements,
    vtk_centroids2surface_measurements,
    get_hollow_mask,
    vtk_voxelizer,
    vtk_is_mesh_closed,
    vtk_is_mesh_manifold,
    vtk_meshes_bbox_sitk_image
)

is_mask_empty = lambda mask: not np.any(mask.astype(bool))

class DistanceMetrics:
    def __init__(self):
        self._ref_np = None
        self._pred_np = None
        self._spacing = None
        self._ref_sitk = None
        self._pred_sitk = None
        self._ref_vtk = None
        self._pred_vtk = None
        self._ref_vtk = None
        self._pred_vtk = None

    def set_input(
        self,
        ref: Union[np.ndarray, sitk.Image, vtk.vtkPolyData],
        pred: Union[np.ndarray, sitk.Image, vtk.vtkPolyData],
        spacing: Union[tuple, list, np.ndarray] = None,
    ):
        """
        General input setter method that automatically detects the input type.
        Note: 
            - If `ref` is numpy array, the `pred` must also be numpy array and vice versa. Also spacing must be set.
            This is to avoid potential issues with different spatial positions of segmentation masks 
            (i.e. mesh in world coordinate system and mask in local coordinate system).
            
            - If both `ref` and `pred` are VTK polydata, the `spacing` must be set.
            This is because some calculations are performed in grid space (see BIoU).
            
            - If both `ref` and `pred` are SimpleITK images, the spacing is automatically set from the images and should not be provided.
            
            - For all other combinations (`ref` is SimpleITK image and `pred` is VTK polydata or vice versa),
            the spacing will be inferred from the SimpleITK image input and should not be provided. 
            The input mesh should be in world coordinates.
        """
        self.clear_cache()
        
        # both np.ndarray
        if isinstance(ref, np.ndarray) or isinstance(pred, np.ndarray):
            assert isinstance(ref, np.ndarray) and isinstance(pred, np.ndarray), "if `ref` is numpy array, pred must also be a numpy array and vice versa"
            assert spacing is not None, "spacing must be provided if either `ref` or `pred` are numpy arrays"
            self._set_input_numpy(ref, pred, spacing)
        # both sitk.Image
        elif isinstance(ref, sitk.Image) and isinstance(pred, sitk.Image):
            assert spacing is None, "spacing must not be provided if both `ref` and `pred` are SimpleITK images"
            self.spacing = ref.GetSpacing()
            self._set_input_SimpleITK(ref, pred)
        # both vtk.vtkPolyData
        elif isinstance(ref, vtk.vtkPolyData) and isinstance(pred, vtk.vtkPolyData):
            assert spacing is not None, "spacing must be provided if both `ref` and `pred` are vtkPolyData"
            self._set_input_vtk(ref, pred, spacing)
        # one is sitk.Image and the other is vtk.vtkPolyData
        elif isinstance(ref, (sitk.Image, vtk.vtkPolyData)) and isinstance(pred, (sitk.Image, vtk.vtkPolyData)):
            assert spacing is None, "spacing will be inferred from the SimpleITK image input"
            if isinstance(ref, sitk.Image):
                self.ref_sitk = ref
                spacing = ref.GetSpacing()
                ref = vtk_meshing(ref)
            else:
                self.pred_sitk = pred
                spacing = pred.GetSpacing()
                pred = vtk_meshing(pred)
            self._set_input_vtk(ref, pred, spacing)
        else:
            assert isinstance(pred, (sitk.Image, vtk.vtkPolyData)), "if `ref` is SimpleITK image, `pred` must be SimpleITK image or vtkPolyData"
            raise ValueError("ref must be a numpy array, SimpleITK image or vtkPolyData")
            
    def _set_input_numpy(
        self,
        ref: np.ndarray,
        pred: np.ndarray,
        spacing: Union[tuple, list, np.ndarray],
    ):
        self.clear_cache()
        assert ref.ndim == pred.ndim == len(spacing), "masks and spacing must all have the same dimensionality"
        assert ref.shape == pred.shape, "masks must have the same shape"

        self.ref_np = ref
        self.pred_np = pred
        self.spacing = spacing

        ## set other representations
        self.ref_sitk = self.ref_np
        self.pred_sitk = self.pred_np
        self.ref_vtk = self.ref_np
        self.pred_vtk = self.pred_np

    def _set_input_SimpleITK(
        self,
        ref: sitk.Image,
        pred: sitk.Image,
    ):
        self.clear_cache()
        assert isinstance(ref, sitk.Image), "ref must be a SimpleITK image"
        assert isinstance(pred, sitk.Image), "pred must be a SimpleITK image"
        assert ref.GetOrigin() == pred.GetOrigin(), "input mask origin must be the same"
        assert ref.GetSpacing() == pred.GetSpacing(), "input mask spacing must be the same"
        assert ref.GetSize() == pred.GetSize(), "input mask size must be the same"
        assert ref.GetDirection() == pred.GetDirection(), "input mask direction must be the same"
        
        self.ref_sitk = ref
        self.pred_sitk = pred
        
        ## set other representations
        self.ref_np = self.ref_sitk
        self.pred_np = self.pred_sitk
        self.ref_vtk = self.ref_sitk
        self.pred_vtk = self.pred_sitk

    def _set_input_vtk(
        self,
        ref: vtk.vtkPolyData,
        pred: vtk.vtkPolyData,
        spacing: Union[tuple, list, np.ndarray],
    ):
        """
        Set the input VTK polydata for reference and prediction meshes along with the spacing.
        Parameters
        ----------
        ref : vtk.vtkPolyData
            The reference mesh as a VTK polydata object.
        pred : vtk.vtkPolyData
            The prediction mesh as a VTK polydata object.
        spacing : Union[tuple, list, np.ndarray]
            The spacing is required, because some calculations are performed in grid space (see BIoU).
        Raises
        ------
        AssertionError
            If `ref` or `pred` are not instances of vtk.vtkPolyData.
            If `ref` or `pred` are not closed meshes.
            If `ref` or `pred` are not manifold meshes.
        """
        self.clear_cache()
        
        self.ref_vtk = ref
        self.pred_vtk = pred
        self.spacing = spacing

        ## set other representations
        # create a meta image SimpleITK that encompasses both masks
        meta_sitk = vtk_meshes_bbox_sitk_image(
            self.ref_vtk, 
            self.pred_vtk, 
            spacing=self.spacing,
            tolerance=5*np.array(self.spacing), 
        )
        
        self.ref_sitk = vtk_voxelizer(self.ref_vtk, meta_sitk)
        self.pred_sitk = vtk_voxelizer(self.pred_vtk, meta_sitk)
        
        self.ref_np = self.ref_sitk
        self.pred_np = self.pred_sitk

    @property
    def n_dim(self):
        return len(self.spacing)

    def clear_cache(self):
        self.__init__()
        cl = self.__class__
        for attr in dir(cl):
            if hasattr(cl, attr):
                cl_attr = getattr(cl, attr)
                if hasattr(cl_attr, 'fget'):
                    cl_attr_fget = getattr(cl_attr, 'fget')
                    if hasattr(cl_attr_fget, 'cache_clear'):
                        cl_attr_fget.cache_clear()
    @property
    def ref_np(self) -> np.ndarray:
        return self._ref_np

    @ref_np.setter
    def ref_np(self, value: np.ndarray):
        if self._ref_np is not None:
            pass
        elif isinstance(value, np.ndarray):
            assert value.dtype == bool, "mask must be a boolean array"
            self._ref_np = value.astype('uint8')
        elif isinstance(value, sitk.Image):
            assert id(value) == id(self.ref_sitk), "mask must be the same object as the `ref_sitk`"
            self._ref_np = sitk2np(value)
        elif isinstance(value, vtk.vtkPolyData):
            raise NotImplementedError("Conversion from vtkPolyData to numpy array is not implemented")
        else:
            raise ValueError("mask must be a numpy array, SimpleITK image or vtkPolyData")

    @property
    def pred_np(self) -> np.ndarray:
        return self._pred_np
    
    @pred_np.setter
    def pred_np(self, value: np.ndarray):
        if self._pred_np is not None:
            pass
        elif isinstance(value, np.ndarray):
            assert value.dtype == bool, "mask must be a boolean array"
            self._pred_np = value.astype('uint8')
        elif isinstance(value, sitk.Image):
            assert id(value) == id(self.pred_sitk), "mask must be the same object as the `pred_sitk`"
            self._pred_np = sitk2np(value)
        elif isinstance(value, vtk.vtkPolyData):
            raise NotImplementedError("Conversion from vtkPolyData to numpy array is not implemented")
        else:
            raise ValueError("mask must be a numpy array, SimpleITK image or vtkPolyData")

    @property
    def spacing(self) -> tuple:
        return self._spacing
    
    @spacing.setter
    def spacing(self, value):
        if self._spacing is not None:
            assert value == self.spacing, "spacing must be the same as the previously set spacing"
        else:
            assert isinstance(
                value, (list, tuple, np.ndarray)
            ), "spacing must be a list, tuple or numpy array"
            assert len(value) in [2, 3], "only 2D or 3D calculations are supported"
            self._spacing = tuple(value)

    @property
    def ref_sitk(self) -> sitk.Image:
        return self._ref_sitk
        
    @ref_sitk.setter
    def ref_sitk(self, value: Union[np.ndarray, sitk.Image, vtk.vtkPolyData]):
        if self._ref_sitk is not None:
            pass
        elif isinstance(value, np.ndarray):
            assert id(value) == id(self.ref_np), "mask must be the same object as the `ref_np`"
            self._ref_sitk = np2sitk(value, spacing=self.spacing)
        elif isinstance(value, sitk.Image):
            assert value.GetPixelID() == sitk.sitkUInt8, "mask must be a sitk image with pixel type UInt8"
            labelimfilter = sitk.LabelShapeStatisticsImageFilter()
            labelimfilter.Execute(value)
            assert labelimfilter.GetNumberOfLabels() < 2, "mask must include background and up to one foreground class"
            self._ref_sitk = value > 0
            self.spacing = value.GetSpacing()
        elif isinstance(value, vtk.vtkPolyData):
            raise NotImplementedError("Conversion from vtkPolyData to SimpleITK image needs to happen in the input setter")
        else:
            raise ValueError("mask must be a numpy array, SimpleITK image or vtkPolyData")
        
    @property
    def pred_sitk(self) -> sitk.Image:
        return self._pred_sitk
        
    @pred_sitk.setter
    def pred_sitk(self, value: Union[sitk.Image, np.ndarray, vtk.vtkPolyData]):
        if self._pred_sitk is not None:
            pass
        elif isinstance(value, np.ndarray):
            assert id(value) == id(self.pred_np), "mask must be the same object as the `pred_np`"
            self._pred_sitk = np2sitk(value, spacing=self.spacing)
        elif isinstance(value, sitk.Image):
            assert value.GetPixelID() == sitk.sitkUInt8, "mask must be a sitk image with pixel type UInt8"
            labelimfilter = sitk.LabelShapeStatisticsImageFilter()
            labelimfilter.Execute(value)
            assert labelimfilter.GetNumberOfLabels() < 2, "mask must include background and up to one foreground class"
            self._pred_sitk = value > 0
        elif isinstance(value, vtk.vtkPolyData):
            raise NotImplementedError("Conversion from vtkPolyData to SimpleITK image needs to happen in the input setter")
        else:
            raise ValueError("mask must be a numpy array, SimpleITK image or vtkPolyData")

    @property
    def ref_vtk(self) -> sitk.Image:
        return self._ref_vtk
        
    @ref_vtk.setter
    def ref_vtk(self, value: Union[np.ndarray, sitk.Image, vtk.vtkPolyData]):
        if self._ref_vtk is not None:
            pass
        elif isinstance(value, np.ndarray):
            assert id(value) == id(self.ref_np), "mask must be the same object as the `ref_np`"
            assert hasattr(self, "ref_sitk"), "ref_sitk must be set before setting the mesh"
            self._ref_vtk = vtk_meshing(self.ref_sitk)
        elif isinstance(value, sitk.Image):
            assert id(value) == id(self.ref_sitk), "mask must be the same object as the `ref_sitk`"
            self._ref_vtk = vtk_meshing(self.ref_sitk)
        elif isinstance(value, vtk.vtkPolyData):
            assert vtk_is_mesh_closed(value), "ref mesh must be a closed mesh"
            assert vtk_is_mesh_manifold(value), "ref mesh must be a manifold mesh"
            self._ref_vtk = value
        else:
            raise ValueError("mask must be a numpy array, SimpleITK image or vtkPolyData")
        
    @property
    def pred_vtk(self) -> sitk.Image:
        return self._pred_vtk
        
    @pred_vtk.setter
    def pred_vtk(self, value: Union[np.ndarray, sitk.Image, vtk.vtkPolyData]):
        if self._pred_vtk is not None:
            pass
        elif isinstance(value, np.ndarray):
            assert id(value) == id(self.pred_np), "mask must be the same object as the `pred_np`"
            assert hasattr(self, "pred_sitk"), "pred_sitk must be set before setting the mesh"
            self._pred_vtk = vtk_meshing(self.pred_sitk)
        elif isinstance(value, sitk.Image):
            assert id(value) == id(self.pred_sitk), "mask must be the same object as the `pred_sitk`"
            self._pred_vtk = vtk_meshing(self.pred_sitk)
        elif isinstance(value, vtk.vtkPolyData):
            assert vtk_is_mesh_closed(value), "pred mesh must be a closed mesh"
            assert vtk_is_mesh_manifold(value), "pred mesh must be a manifold mesh"
            self._pred_vtk = value
        else:
            raise ValueError("mask must be a numpy array, SimpleITK image or vtkPolyData")
        
    @property
    @lru_cache
    def ref_is_empty(self) -> bool:
        return self.ref_vtk.GetNumberOfPoints() == 0

    @property
    @lru_cache
    def pred_is_empty(self) -> bool:
        return self.pred_vtk.GetNumberOfPoints() == 0

    @property
    @lru_cache
    def distances(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.ref_is_empty or self.pred_is_empty:
            return None
        elif self.n_dim == 2:
            d_ref2pred, b_ref, d_pred2ref, b_pred = vtk_centroids2contour_measurements(
            ref_contour=self.ref_vtk,
            pred_contour=self.pred_vtk,
            subdivide_iter=5,
        )
        elif self.n_dim == 3:
            d_ref2pred, b_ref, d_pred2ref, b_pred = vtk_centroids2surface_measurements(
            ref_mesh=self.ref_vtk, 
            pred_mesh=self.pred_vtk,
            subdivide_iter=1
        )
        else:
            raise ValueError("Only 2D and 3D masks are supported")

        return d_ref2pred, b_ref, d_pred2ref, b_pred

    @staticmethod
    def perc_surface_dist(dists, b_sizes, perc) -> float:
        if len(dists) > 0:
            cum_surfel_areas = np.cumsum(b_sizes) / np.sum(b_sizes)
            idx = np.searchsorted(cum_surfel_areas, perc / 100.0)
            idx = min(idx, len(dists) - 1)
            return dists[idx]
        else:
            return np.inf

    def hd(self, percentile=100) -> float:
        assert isinstance(percentile, int), "percentile must be an integer"
        assert 0 <= percentile <= 100, "percentile must be between 0 and 100"

        if self.ref_is_empty and self.pred_is_empty:
            return np.nan
        elif self.ref_is_empty or self.pred_is_empty:
            return np.inf
        else:
            d_ref2pred, b_ref, d_pred2ref, b_pred = self.distances
            perc_d_ref2pred = self.perc_surface_dist(d_ref2pred, b_ref, percentile)
            perc_d_pred2ref = self.perc_surface_dist(d_pred2ref, b_pred, percentile)
            return max(perc_d_ref2pred, perc_d_pred2ref)

    def masd(self) -> float:
        if self.ref_is_empty and self.pred_is_empty:
            logging.warning("Both masks are empty")
            return np.nan
        elif self.ref_is_empty or self.pred_is_empty:
            logging.warning("One of the masks is empty")
            return np.inf
        else:
            d_ref2pred, b_ref, d_pred2ref, b_pred = self.distances
            d_ref2pred = (d_ref2pred @ b_ref) / b_ref.sum()  # change to * operator
            d_pred2ref = (d_pred2ref @ b_pred) / b_pred.sum()  # change to * operator
            return (d_ref2pred + d_pred2ref) / 2

    def assd(self) -> float:
        if self.ref_is_empty and self.pred_is_empty:
            return np.nan
        elif self.ref_is_empty or self.pred_is_empty:
            logging.warning("One of the masks is empty")
            return np.inf
        else:
            d_ref2pred, b_ref, d_pred2ref, b_pred = self.distances
            num = d_ref2pred @ b_ref + d_pred2ref @ b_pred
            denom = b_ref.sum() + b_pred.sum()
            if denom == 0:
                raise ValueError("sum of boundary sizes is zero, something weird is going on")
            value = num / denom
            return value

    def nsd(self, tau) -> float:
        assert isinstance(tau, (int, float)), "tolerance must be a float"
        assert tau >= 0, "tolerance must be greater than or equal to zero"

        if self.ref_is_empty and self.pred_is_empty:
            logging.warning("Both masks are empty")
            return np.nan
        elif self.ref_is_empty or self.pred_is_empty:
            logging.warning("One of the masks is empty")
            return 0
        else:
            d_ref2pred, b_ref, d_pred2ref, b_pred = self.distances
            overlap_ref = b_ref[d_ref2pred <= tau].sum()
            overlap_pred = b_pred[d_pred2ref <= tau].sum()
            num = overlap_ref + overlap_pred
            denom = b_ref.sum() + b_pred.sum()
            if denom == 0:
                raise ValueError("sum of boundary sizes is zero, something weird is going on")
            return num / denom

    def biou(self, tau) -> float:
        assert isinstance(tau, (int, float)), "tolerance must be a float"
        assert tau > 0, "tolerance must be greater than or equal to zero"

        if self.ref_is_empty and self.pred_is_empty:
            logging.warning("Both masks are empty")
            return np.nan
        elif self.ref_is_empty or self.pred_is_empty:
            logging.warning("One of the masks is empty")
            return 0
        else:
            ref_hollow_np, pred_hollow_np = get_hollow_mask(
                ref_mask=self.ref_sitk, 
                pred_mask=self.pred_sitk,
                tau=tau
            )
            
            num = np.logical_and(ref_hollow_np, pred_hollow_np).sum()
            denom = np.logical_or(ref_hollow_np, pred_hollow_np).sum()

            return num / denom
