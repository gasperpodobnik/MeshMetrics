from functools import lru_cache
import logging
from typing import Any, Union

import numpy as np
from .mesh_utils import (
    vtk_2D_meshing,
    vtk_3D_meshing,
    np2sitk,
    sitk_add_axis,
    vtk_centroids2contour_measurements,
    vtk_centroids2surface_measurements,
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
    def ref_mask(self):
        return self._ref_mask

    @ref_mask.setter
    def ref_mask(self, value):
        assert isinstance(value, np.ndarray), "mask must be a numpy array"
        assert value.dtype == bool, "mask must be a boolean array"
        self._ref_mask = value.astype('uint8')

    @property
    def pred_mask(self):
        return self._pred_mask

    @pred_mask.setter
    def pred_mask(self, value):
        assert isinstance(value, np.ndarray), "mask must be a numpy array"
        assert value.dtype == bool, "mask must be a boolean array"
        self._pred_mask = value.astype('uint8')

    @property
    def spacing(self):
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
    def ref_mesh(self):
        ref_sitk_3D = np2sitk(self.ref_mask, spacing=self.spacing)
        return vtk_3D_meshing(ref_sitk_3D)
        
    @property
    @lru_cache
    def pred_mesh(self):
        pred_sitk_3D = np2sitk(self.pred_mask, spacing=self.spacing)
        return vtk_3D_meshing(pred_sitk_3D)

    # fmt: off
    @property
    @lru_cache
    def distances(self) -> dict:
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

    def _distances_2D(self):
        ref_sitk_2D = np2sitk(self.ref_mask, spacing=self.spacing)
        pred_sitk_2D = np2sitk(self.pred_mask, spacing=self.spacing)

        ref_contour = vtk_2D_meshing(ref_sitk_2D)
        pred_contour = vtk_2D_meshing(pred_sitk_2D)

        ref_sitk_3D = sitk_add_axis(ref_sitk_2D)
        pred_sitk_3D = sitk_add_axis(pred_sitk_2D)

        ref_surface = vtk_3D_meshing(ref_sitk_3D)
        pred_surface = vtk_3D_meshing(pred_sitk_3D)

        return vtk_centroids2contour_measurements(
            ref_contour=ref_contour,
            ref_surface=ref_surface,
            pred_contour=pred_contour,
            pred_surface=pred_surface,
            subdivide_iter=5,
        )

    def _distances_3D(self):
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

    def hd(self, percentile=100):
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

    def masd(self):
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

    def assd(self):
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

    def nsd(self, tau):
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

    def biou(self, tau):
        assert isinstance(tau, (int, float)), "tolerance must be a float"
        assert tau > 0, "tolerance must be greater than or equal to zero"
        
        from .mesh_utils import get_hollow_meshes

        if self.ref_mask.sum() == 0 and self.pred_mask.sum() == 0:
            logging.warning("Both masks are empty")
            return np.nan
        elif self.ref_mask.sum() == 0 or self.pred_mask.sum() == 0:
            logging.warning("One of the masks is empty")
            return 0
        else:
            if self.n_dim == 2:
                raise NotImplementedError("Boundary IoU is not yet implemented for 2D masks")
            elif self.n_dim == 3:
                ref_hollow_np, pred_hollow_np = get_hollow_meshes(self.ref_mesh, self.pred_mesh, spacing=self.spacing, tau=tau)

            num = np.logical_and(ref_hollow_np, pred_hollow_np).sum()
            denom = np.logical_or(ref_hollow_np, pred_hollow_np).sum()

            return num / denom
