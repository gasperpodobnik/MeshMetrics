from pathlib import Path
from typing import Tuple, Union

import numpy as np
import SimpleITK as sitk
import logging

from SimpleITK.utilities.vtk import sitk2vtk, vtk2sitk
import vtk
from vtk.util.numpy_support import vtk_to_numpy


def np2sitk(img_np: np.ndarray, spacing, swapaxes=True) -> sitk.Image:
    if swapaxes:
        assert img_np.ndim in [2, 3], "Unsupported number of dimensions"
        img_np = np.swapaxes(img_np, 0, -1)
    img_sitk = sitk.GetImageFromArray(img_np)
    img_sitk.SetSpacing(spacing)
    return img_sitk


def sitk2np(sitk_img: sitk.Image) -> np.ndarray:
    assert sitk_img.GetDimension() in [2, 3], "Unsupported number of dimensions"
    return np.swapaxes(sitk.GetArrayFromImage(sitk_img), 0, -1)


def sitk_add_axis(img_sitk_2d, thickness):
    ref_np_3d = sitk2np(img_sitk_2d)[..., np.newaxis]
    spacing = (*img_sitk_2d.GetSpacing(), thickness)
    img_sitk_3d = np2sitk(ref_np_3d, spacing=spacing)
    img_sitk_3d.SetOrigin((*img_sitk_2d.GetOrigin(), 0.0))
    return img_sitk_3d


def to_sitk(img: Union[str, Path, sitk.Image]) -> sitk.Image:
    if isinstance(img, (str, Path)):
        img = sitk.ReadImage(str(img))
    elif isinstance(img, sitk.Image):
        pass
    else:
        raise NotImplementedError(f"Unknown image type: {type(img)}")
    return img


def vtk_read_polydata(pth: Union[str, Path]) -> vtk.vtkPolyData:
    reader = vtk.vtkOBJReader()
    reader.SetFileName(str(pth))
    reader.Update()
    polydata = reader.GetOutput()
    return polydata


def to_vtk(src_mesh: Union[str, Path, vtk.vtkPolyData]) -> vtk.vtkPolyData:
    if isinstance(src_mesh, (str, Path)):
        src_mesh = vtk_read_polydata(src_mesh)
    elif isinstance(src_mesh, vtk.vtkPolyData):
        pass
    else:
        raise NotImplementedError(f"Unknown mesh type: {type(src_mesh)}")
    return src_mesh


def vtk_is_mesh_closed(polydata):
    """
    Check if a surface is closed (i.e., it has no boundary edges).

    Parameters:
    polydata (vtk.vtkPolyData): The input surface mesh.

    Returns:
    bool: True if the surface is closed, False otherwise.
    """
    polydata = to_vtk(polydata)
    # Initialize the vtkFeatureEdges filter
    feature_edges = vtk.vtkFeatureEdges()
    feature_edges.SetInputData(polydata)
    feature_edges.BoundaryEdgesOn()
    feature_edges.FeatureEdgesOff()
    feature_edges.ManifoldEdgesOff()
    feature_edges.NonManifoldEdgesOff()
    feature_edges.Update()

    # Get the number of boundary edges
    boundary_edges = feature_edges.GetOutput().GetNumberOfCells()

    return boundary_edges == 0


def vtk_is_mesh_manifold(polydata):
    """
    Check if a surface is manifold
    (i.e., every mesh edge is shared by at most two faces).

    Parameters:
    polydata (vtk.vtkPolyData): The input surface mesh.

    Returns:
    bool: True if the surface is closed, False otherwise.
    """

    polydata = to_vtk(polydata)
    # Initialize the vtkFeatureEdges filter
    feature_edges = vtk.vtkFeatureEdges()
    feature_edges.SetInputData(polydata)
    feature_edges.BoundaryEdgesOff()
    feature_edges.FeatureEdgesOff()
    feature_edges.ManifoldEdgesOff()
    feature_edges.NonManifoldEdgesOn()
    feature_edges.Update()

    # Get the number of boundary edges
    boundary_edges = feature_edges.GetOutput().GetNumberOfCells()

    return boundary_edges == 0


def vtk_2D_meshing(src_img: Union[str, Path, sitk.Image]) -> vtk.vtkPolyData:
    src_img = to_sitk(src_img)

    n_dim = src_img.GetDimension()
    assert n_dim == 2, "Only 2D images are supported for marching squares"

    if sitk.GetArrayFromImage(src_img).sum() == 0:
        return None

    # pad to avoid potential open boundary related issues
    src_img = sitk.ConstantPad(src_img, (1, 1), (1, 1), 0)

    vtkImage = sitk2vtk(src_img > 0)

    meshing_alg = vtk.vtkDiscreteFlyingEdges2D()
    meshing_alg.SetInputData(vtkImage)
    meshing_alg.Update()
    mesh = meshing_alg.GetOutput()

    assert vtk_is_mesh_closed(mesh), "Mesh is not closed"

    return mesh


def vtk_3D_meshing(src_img: Union[str, Path, sitk.Image]) -> vtk.vtkPolyData:
    src_img = to_sitk(src_img)

    n_dim = src_img.GetDimension()
    assert n_dim == 3, "Only 3D images are supported for marching cubes"

    # pad to avoid potential open boundary related issues
    src_img = sitk.ConstantPad(src_img, (1, 1, 1), (1, 1, 1), 0)

    if sitk.GetArrayFromImage(src_img).sum() == 0:
        return None
    vtkImage = sitk2vtk(src_img > 0)

    # segmentation --> polygonal data:
    meshing_alg = vtk.vtkDiscreteMarchingCubes()
    meshing_alg.ComputeNormalsOn()
    meshing_alg.SetInputData(vtkImage)
    meshing_alg.Update()
    mesh = meshing_alg.GetOutput()

    assert vtk_is_mesh_closed(mesh), "Mesh is not closed"

    return mesh


def vtk_meshing(src_img: Union[str, Path, sitk.Image]):
    src_img = to_sitk(src_img)

    n_dim = src_img.GetDimension()
    if n_dim == 2:
        mesh = vtk_2D_meshing(src_img)
    elif n_dim == 3:
        mesh = vtk_3D_meshing(src_img)
    else:
        raise ValueError("Only 2D or 3D images are supported")
    return mesh


def vtk_2D_centroid2surface_dist_length(
    pts_contour: vtk.vtkPolyData,
    surface_mesh: vtk.vtkPolyData,
    subdivide_iter: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:

    # compute distances and face areas between ref centroids and pred mesh
    N_segments = pts_contour.GetNumberOfLines()
    dists_pts2surface, segment_lengths = np.zeros(
        N_segments * (subdivide_iter + 1)
    ), np.zeros(N_segments * (subdivide_iter + 1))

    vtk_p2s_dist = vtk.vtkImplicitPolyDataDistance()
    vtk_p2s_dist.SetInput(surface_mesh)
    cnt = 0
    for enum in range(N_segments):
        pts = vtk_to_numpy(pts_contour.GetCell(enum).GetPoints().GetData())
        assert pts.shape[0] == 2, "Only 2D segments are supported"
        vec = pts[1] - pts[0]
        ks = np.linspace(0, 1, 2 + subdivide_iter)
        new_pts = np.vstack([pts[0] + k * vec for k in ks])
        for enum2 in range(len(new_pts) - 1):
            pt0 = new_pts[enum2]
            pt1 = new_pts[enum2 + 1]
            segment_lengths[cnt] = np.linalg.norm(pt0 - pt1)
            dists_pts2surface[cnt] = abs(vtk_p2s_dist.FunctionValue((pt0 + pt1) / 2))
            cnt += 1
    return dists_pts2surface, segment_lengths


def sort_dists_and_bsizes(dists, boundary_sizes) -> Tuple[np.ndarray, np.ndarray]:
    _sorted = np.array(sorted(zip(dists, boundary_sizes)))
    dists_s, boundary_sizes_s = _sorted[:, 0], _sorted[:, 1]
    return dists_s, boundary_sizes_s


def vtk_centroids2contour_measurements(
    ref_contour: vtk.vtkPolyData,
    ref_surface: vtk.vtkPolyData,
    pred_contour: vtk.vtkPolyData,
    pred_surface: vtk.vtkPolyData,
    subdivide_iter: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate distances between centroids of triangles and surface represented by the mesh.
    Function calculates in both directions and also returns the areas of surface elements (surfels).

    Args:
        ref_mesh (Union[str, Path, trimesh.Trimesh]): _description_
        pred_mesh (Union[str, Path, trimesh.Trimesh]): _description_

    Returns:
        tuple: (numpy vector of distances between ref mesh vertices and pred mesh surface, numpy vector of distances between pred mesh vertices and ref mesh surface)
    """

    # fmt: off
    dists_ref2pred, segment_lengths_ref = vtk_2D_centroid2surface_dist_length(ref_contour, pred_surface, subdivide_iter=subdivide_iter)
    dists_pred2ref, segment_lengths_pred = vtk_2D_centroid2surface_dist_length(pred_contour, ref_surface, subdivide_iter=subdivide_iter)
    
    # sort distances and boundary sizes
    dists_ref2pred, segment_lengths_ref = sort_dists_and_bsizes(dists_ref2pred, segment_lengths_ref)
    dists_pred2ref, segment_lengths_pred = sort_dists_and_bsizes(dists_pred2ref, segment_lengths_pred)
    # fmt: on

    return dists_ref2pred, segment_lengths_ref, dists_pred2ref, segment_lengths_pred


def vtk_subdivide_mesh(vtk_polydata: vtk.vtkPolyData) -> vtk.vtkPolyData:
    # note: does not work for meshes created with `surface_nets`
    vtk_subd = vtk.vtkLinearSubdivisionFilter()
    vtk_subd.SetInputData(vtk_polydata)
    vtk_subd.Update()
    return vtk_subd.GetOutput()


def vtk_compute_cell_sizes(mesh: vtk.vtkPolyData) -> np.ndarray:
    N_faces = mesh.GetNumberOfCells()
    cell_sizes = np.zeros(N_faces)
    for enum in range(N_faces):
        cell_sizes[enum] = mesh.GetCell(enum).ComputeArea()
    return cell_sizes


def vtk_centroids2surface_measurements(
    ref_mesh: vtk.vtkPolyData,
    pred_mesh: vtk.vtkPolyData,
    subdivide_iter: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate distances between centroids of triangles and surface represented by the mesh.
    Function calculates in both directions and also returns the areas of surface elements (surfels).

    Args:
        ref_mesh (Union[str, Path, trimesh.Trimesh]): _description_
        pred_mesh (Union[str, Path, trimesh.Trimesh]): _description_

    Returns:
        tuple: (numpy vector of distances between ref mesh vertices and pred mesh surface, numpy vector of distances between pred mesh vertices and ref mesh surface)
    """

    ref_manifold = vtk_is_mesh_manifold(ref_mesh)
    pred_manifold = vtk_is_mesh_manifold(pred_mesh)

    if ref_manifold and pred_manifold:
        for _ in range(subdivide_iter):
            ref_mesh = vtk_subdivide_mesh(ref_mesh)
            pred_mesh = vtk_subdivide_mesh(pred_mesh)
    else:
        if subdivide_iter > 0:
            logging.warning("Meshes are not manifold, skipping subdivision")

    vtk_p2v_dist = vtk.vtkDistancePolyDataFilter()
    vtk_p2v_dist.SetInputData(0, ref_mesh)
    vtk_p2v_dist.SetInputData(1, pred_mesh)
    vtk_p2v_dist.SignedDistanceOff()
    vtk_p2v_dist.ComputeCellCenterDistanceOn()
    vtk_p2v_dist.ComputeSecondDistanceOn()
    vtk_p2v_dist.Update()
    dists_ref2pred = vtk_to_numpy(
        vtk_p2v_dist.GetOutput().GetCellData().GetArray("Distance")
    )
    dists_pred2ref = vtk_to_numpy(
        vtk_p2v_dist.GetSecondDistanceOutput().GetCellData().GetArray("Distance")
    )
    surfel_areas_ref = vtk_compute_cell_sizes(ref_mesh)
    surfel_areas_pred = vtk_compute_cell_sizes(pred_mesh)

    # fmt: off
    # sort distances and boundary sizes
    dists_ref2pred, surfel_areas_ref = sort_dists_and_bsizes(dists_ref2pred, surfel_areas_ref)
    dists_pred2ref, surfel_areas_pred = sort_dists_and_bsizes(dists_pred2ref, surfel_areas_pred)
    # fmt: on

    return dists_ref2pred, surfel_areas_ref, dists_pred2ref, surfel_areas_pred


def vtk_compute_normals(mesh: vtk.vtkPolyData) -> vtk.vtkPolyData:
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(mesh)
    normals.ComputePointNormalsOn()  # Optional: compute point normals
    normals.ComputeCellNormalsOn()  # Optional: compute cell normals
    normals.Update()
    mesh_normals = normals.GetOutput()
    return mesh_normals


def vtk_signed_distance(vtk_mesh, dist, bounds, size):
    distance_filter = vtk.vtkSignedDistance()
    distance_filter.SetInputData(vtk_mesh)
    distance_filter.SetRadius(dist)  # Radius to compute signed distance within
    distance_filter.SetBounds(bounds)
    distance_filter.SetDimensions(size)
    distance_filter.Update()
    distance_field_sitk = vtk2sitk(distance_filter.GetOutput())
    return distance_field_sitk

def implicit_signed_distance(vtk_mesh, origin, diagonal, spacing):
    ox, oy, oz = origin
    dx, dy, dz = diagonal
    sx, sy, sz = spacing
    
    # stack meshgrid
    pts = np.stack(np.meshgrid(np.arange(ox, dx+sx, sx), np.arange(oy, dy+sy, sy), np.arange(oz, dz+sz, sz), indexing='ij'), axis=-1)
    # flatten in a list of 3d points
    pts_reshaped = pts.reshape(-1, 3)
    out = np.empty(pts_reshaped.shape[0])
    
    vtk_p2s_dist = vtk.vtkImplicitPolyDataDistance()
    vtk_p2s_dist.SetInput(vtk_mesh)
    for enum, pt in enumerate(pts_reshaped):
        out[enum] = vtk_p2s_dist.FunctionValue(pt)
    out = out.reshape(pts.shape[:-1])
    
    distance_field = np2sitk(out, spacing)
    distance_field.SetOrigin(origin)
    return distance_field

def get_boundary_region(
    ref_mesh: vtk.vtkPolyData, pred_mesh: vtk.vtkPolyData, spacing: tuple, tau: float
):
    assert tau > 0, "Distance must be positive"
    
    r_b = np.array(ref_mesh.GetBounds())
    p_b = np.array(pred_mesh.GetBounds())
    ref_origin, ref_diagonal =r_b[::2], r_b[1::2]
    pred_origin, pred_diagonal = p_b[::2], p_b[1::2]
    
    # find element-wise minimum and maximum
    _origin = np.minimum(ref_origin, pred_origin)
    _diagonal = np.maximum(ref_diagonal, pred_diagonal)

    _spacing = np.array(spacing) / 5
    if len(spacing) == 2:
        _spacing = np.append(_spacing, 1.0)
        _origin[-1] = 0
        _diagonal[-1] = 0
        
    ref_dist_field_sitk = implicit_signed_distance(ref_mesh, _origin, _diagonal, _spacing)
    pred_dist_field_sitk = implicit_signed_distance(pred_mesh, _origin, _diagonal, _spacing)

    # get hollowed masks
    sitk_and = sitk.AndImageFilter()
    ref_hollowed_seg_sitk = sitk_and.Execute(ref_dist_field_sitk < tau, ref_dist_field_sitk >= 0)
    pred_hollowed_seg_sitk = sitk_and.Execute(pred_dist_field_sitk < tau, pred_dist_field_sitk >= 0)
    
    return sitk2np(ref_hollowed_seg_sitk) > 0, sitk2np(pred_hollowed_seg_sitk) > 0
