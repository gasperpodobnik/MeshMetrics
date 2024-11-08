from functools import lru_cache
import logging
from typing import Any, Tuple, Union

import numpy as np
import vtk
from .mesh_utils import (
    vtk_meshing,
    np2sitk,
    sitk_add_axis,
    vtk_centroids2contour_measurements,
    vtk_centroids2surface_measurements,
    get_boundary_region,
)

is_mask_empty = lambda mask: not np.any(mask.astype(bool))


class DistanceMetrics:
    def set_input(
        self,
        ref_mask: np.ndarray[Any, np.dtype[np.bool_]],
        pred_mask: np.ndarray[Any, np.dtype[np.bool_]],
        spacing: Union[tuple, list, np.ndarray],
    ):
        self.clear_cache()
        self.ref_mask = ref_mask
        self.pred_mask = pred_mask
        self.spacing = spacing

        # fmt: off
        assert self.ref_mask.ndim == self.pred_mask.ndim == len(spacing), "masks and spacing must all have the same dimensionality"
        assert self.ref_mask.shape == self.pred_mask.shape, "masks must have the same shape"
        # fmt: on

        self.n_dim = len(spacing)

    def clear_cache(self):
        cl = self.__class__
        for attr in dir(cl):
            if hasattr(cl, attr):
                cl_attr = getattr(cl, attr)
                if hasattr(cl_attr, 'fget'):
                    cl_attr_fget = getattr(cl_attr, 'fget')
                    if hasattr(cl_attr_fget, 'cache_clear'):
                        cl_attr_fget.cache_clear()
    @property
    def ref_mask(self) -> np.ndarray:
        return self._ref_mask

    @ref_mask.setter
    def ref_mask(self, value):
        assert isinstance(value, np.ndarray), "mask must be a numpy array"
        assert value.dtype == bool, "mask must be a boolean array"
        self._ref_mask = value.astype('uint8')

    @property
    def pred_mask(self) -> np.ndarray:
        return self._pred_mask

    @pred_mask.setter
    def pred_mask(self, value):
        assert isinstance(value, np.ndarray), "mask must be a numpy array"
        assert value.dtype == bool, "mask must be a boolean array"
        self._pred_mask = value.astype('uint8')

    @property
    def spacing(self) -> tuple:
        return self._spacing
    
    @spacing.setter
    def spacing(self, value):
        assert isinstance(
            value, (list, tuple, np.ndarray)
        ), "spacing must be a list, tuple or numpy array"
        assert len(value) in [2, 3], "only 2D or 3D spacing is supported"
        self._spacing = tuple(value)
        
        
    @property
    @lru_cache
    def ref_mesh(self) -> vtk.vtkPolyData:
        ref_sitk = np2sitk(self.ref_mask, spacing=self.spacing)
        return vtk_meshing(ref_sitk)
        
    @property
    @lru_cache
    def pred_mesh(self) -> vtk.vtkPolyData:
        pred_sitk = np2sitk(self.pred_mask, spacing=self.spacing)
        return vtk_meshing(pred_sitk)
    
    @property
    @lru_cache
    def auxiliary_surface_meshes_for_2d(self) -> Tuple[vtk.vtkPolyData, vtk.vtkPolyData]:
        r_b = np.array(self.ref_mesh.GetBounds())
        p_b = np.array(self.pred_mesh.GetBounds())
        ref_origin, ref_diagonal =r_b[::2], r_b[1::2]
        pred_origin, pred_diagonal = p_b[::2], p_b[1::2]
        
        # find element-wise minimum and maximum
        _origin = np.minimum(ref_origin, pred_origin)
        _diagonal = np.maximum(ref_diagonal, pred_diagonal)
        slice_thickness = np.linalg.norm(_diagonal - _origin)*2
        
        # create surface meshes (surface nets is no good, because signed distance map is not available)
        ref_surface = vtk_meshing(sitk_add_axis(np2sitk(self.ref_mask, spacing=self.spacing), slice_thickness))
        pred_surface = vtk_meshing(sitk_add_axis(np2sitk(self.pred_mask, spacing=self.spacing), slice_thickness))
        return ref_surface, pred_surface

    # fmt: off
    @property
    @lru_cache
    def distances(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.ref_mask.sum() == 0 or self.pred_mask.sum() == 0:
            return None
        elif self.n_dim == 2:
            d_ref2pred, b_ref, d_pred2ref, b_pred = self._distances_2D()
        elif self.n_dim == 3:
            d_ref2pred, b_ref, d_pred2ref, b_pred =  self._distances_3D()
        else:
            raise ValueError("Only 2D and 3D masks are supported")

        return d_ref2pred, b_ref, d_pred2ref, b_pred
    # fmt: on

    def _distances_2D(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ref_contour, pred_contour = self.ref_mesh, self.pred_mesh
        ref_surface, pred_surface = self.auxiliary_surface_meshes_for_2d

        return vtk_centroids2contour_measurements(
            ref_contour=ref_contour,
            ref_surface=ref_surface,
            pred_contour=pred_contour,
            pred_surface=pred_surface,
            subdivide_iter=5,
        )

    def _distances_3D(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return vtk_centroids2surface_measurements(self.ref_mesh, self.pred_mesh, subdivide_iter=1)

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

        if self.ref_mask.sum() == 0 and self.pred_mask.sum() == 0:
            return np.nan
        elif self.ref_mask.sum() == 0 or self.pred_mask.sum() == 0:
            return np.inf
        else:
            d_ref2pred, b_ref, d_pred2ref, b_pred = self.distances
            perc_d_ref2pred = self.perc_surface_dist(d_ref2pred, b_ref, percentile)
            perc_d_pred2ref = self.perc_surface_dist(d_pred2ref, b_pred, percentile)
            return max(perc_d_ref2pred, perc_d_pred2ref)

    def masd(self) -> float:
        if self.ref_mask.sum() == 0 and self.pred_mask.sum() == 0:
            logging.warning("Both masks are empty")
            return np.nan
        elif self.ref_mask.sum() == 0 or self.pred_mask.sum() == 0:
            logging.warning("One of the masks is empty")
            return np.inf
        else:
            d_ref2pred, b_ref, d_pred2ref, b_pred = self.distances
            d_ref2pred = (d_ref2pred @ b_ref) / b_ref.sum()  # change to * operator
            d_pred2ref = (d_pred2ref @ b_pred) / b_pred.sum()  # change to * operator
            return (d_ref2pred + d_pred2ref) / 2

    def assd(self) -> float:
        if self.ref_mask.sum() == 0 and self.pred_mask.sum() == 0:
            return np.nan
        elif self.ref_mask.sum() == 0 or self.pred_mask.sum() == 0:
            logging.warning("One of the masks is empty")
            return np.inf
        else:
            d_ref2pred, b_ref, d_pred2ref, b_pred = self.distances
            num = d_ref2pred @ b_ref + d_pred2ref @ b_pred
            denom = b_ref.sum() + b_pred.sum()
            if denom == 0:
                # fmt: off
                raise ValueError("sum of boundary sizes is zero, something weird is going on")
            # fmt: on
            value = num / denom
            return value

    def nsd(self, tau) -> float:
        assert isinstance(tau, (int, float)), "tolerance must be a float"
        assert tau >= 0, "tolerance must be greater than or equal to zero"

        if self.ref_mask.sum() == 0 and self.pred_mask.sum() == 0:
            logging.warning("Both masks are empty")
            return np.nan
        elif self.ref_mask.sum() == 0 or self.pred_mask.sum() == 0:
            logging.warning("One of the masks is empty")
            return 0
        else:
            d_ref2pred, b_ref, d_pred2ref, b_pred = self.distances
            overlap_ref = b_ref[d_ref2pred <= tau].sum()
            overlap_pred = b_pred[d_pred2ref <= tau].sum()
            num = overlap_ref + overlap_pred
            denom = b_ref.sum() + b_pred.sum()
            if denom == 0:
                # fmt: off
                raise ValueError("sum of boundary sizes is zero, something weird is going on")
            # fmt: on
            return num / denom

    def biou(self, tau) -> float:
        assert isinstance(tau, (int, float)), "tolerance must be a float"
        assert tau > 0, "tolerance must be greater than or equal to zero"

        if self.ref_mask.sum() == 0 and self.pred_mask.sum() == 0:
            logging.warning("Both masks are empty")
            return np.nan
        elif self.ref_mask.sum() == 0 or self.pred_mask.sum() == 0:
            logging.warning("One of the masks is empty")
            return 0
        else:
            if self.n_dim == 2:
                ref_surface, pred_surface = self.auxiliary_surface_meshes_for_2d
                ref_hollow_np, pred_hollow_np = get_boundary_region(ref_surface, pred_surface, spacing=self.spacing, tau=tau)
            elif self.n_dim == 3:
                ref_hollow_np, pred_hollow_np = get_boundary_region(self.ref_mesh, self.pred_mesh, spacing=self.spacing, tau=tau)

            num = np.logical_and(ref_hollow_np, pred_hollow_np).sum()
            denom = np.logical_or(ref_hollow_np, pred_hollow_np).sum()

            return num / denom
