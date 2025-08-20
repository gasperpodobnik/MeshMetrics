from pathlib import Path
from typing import Tuple, Union

import numpy as np
import SimpleITK as sitk
import vtk

from vtk.util.numpy_support import vtk_to_numpy
from SimpleITK.utilities.vtk import sitk2vtk, vtk2sitk


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


def sitk_add_axis(img_sitk_2d: sitk.Image, thickness: float) -> sitk.Image:
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


def vtk_write_polydata(vtk_polydata: vtk.vtkPolyData, dst_pth: Union[str, Path]):
    writer = vtk.vtkOBJWriter()
    writer.SetInputData(vtk_polydata)
    writer.SetFileName(str(dst_pth))
    writer.Write()


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
        return vtk.vtkPolyData()

    vtkImage = sitk2vtk(src_img > 0)

    vtk.vtkLogger.SetStderrVerbosity(vtk.vtkLogger.VERBOSITY_OFF)
    meshing_alg = vtk.vtkSurfaceNets2D()
    meshing_alg.SmoothingOff()
    meshing_alg.SetInputData(vtkImage)
    meshing_alg.Update()
    mesh = meshing_alg.GetOutput()

    return mesh


def vtk_3D_meshing(src_img: Union[str, Path, sitk.Image]) -> vtk.vtkPolyData:
    src_img = to_sitk(src_img)

    n_dim = src_img.GetDimension()
    assert n_dim == 3, "Only 3D images are supported for marching cubes"

    if sitk.GetArrayViewFromImage(src_img).sum() == 0:
        return vtk.vtkPolyData()

    # vtkDiscreteMarchingCubes normals point inward by default, so we invert the image
    vtkImage = sitk2vtk(src_img)

    # segmentation --> polygonal data:

    vtk.vtkLogger.SetStderrVerbosity(vtk.vtkLogger.VERBOSITY_OFF)
    meshing_alg = vtk.vtkSurfaceNets3D()
    meshing_alg.SmoothingOff()
    meshing_alg.SetOutputMeshTypeToTriangles()
    meshing_alg.SetInputData(vtkImage)
    meshing_alg.Update()
    mesh = meshing_alg.GetOutput()

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


def vtkLinearSubdividePolyline(polyline: vtk.vtkPolyData) -> vtk.vtkPolyData:

    points = polyline.GetPoints()
    lines = polyline.GetLines()

    # Create new points and connectivity for the subdivided polyline
    new_points = vtk.vtkPoints()
    new_lines = vtk.vtkCellArray()

    # Traverse the original polyline and add midpoints
    lines.InitTraversal()
    id_list = vtk.vtkIdList()
    while lines.GetNextCell(id_list):
        for i in range(id_list.GetNumberOfIds() - 1):
            # Get the two endpoints of the current segment
            p0 = np.array(points.GetPoint(id_list.GetId(i)))
            p1 = np.array(points.GetPoint(id_list.GetId(i + 1)))

            # Add the first endpoint to the new points
            id0 = new_points.InsertNextPoint(p0)

            # Compute and add the midpoint
            midpoint = (p0 + p1) / 2
            id_mid = new_points.InsertNextPoint(midpoint)

            # Add two new line segments
            new_lines.InsertNextCell(2)
            new_lines.InsertCellPoint(id0)
            new_lines.InsertCellPoint(id_mid)

        # Add the last point of the segment
        id_last = new_points.InsertNextPoint(
            points.GetPoint(id_list.GetId(id_list.GetNumberOfIds() - 1))
        )
        new_lines.InsertNextCell(2)
        new_lines.InsertCellPoint(id_mid)
        new_lines.InsertCellPoint(id_last)

    # Create a new polyline with subdivided segments
    subdivided_polyline = vtk.vtkPolyData()
    subdivided_polyline.SetPoints(new_points)
    subdivided_polyline.SetLines(new_lines)
    return subdivided_polyline


def vtk_2D_centroid2surface_dist_length(
    pts_contour: vtk.vtkPolyData,
    surface_mesh: vtk.vtkPolyData,
) -> Tuple[np.ndarray, np.ndarray]:

    lines = pts_contour.GetLines()
    lines.InitTraversal()
    id_list = vtk.vtkIdList()

    assert (
        lines.GetMaxCellSize() == 2
    ), "This function supports segment lines that have 2 points"

    N = lines.GetNumberOfCells()
    dists_pts2surface, segment_lengths = np.zeros(N), np.zeros(N)

    vtk_p2s_dist = vtk.vtkImplicitPolyDataDistance()
    vtk_p2s_dist.SetInput(surface_mesh)

    # Iterate over each polyline in vtkPolyData
    cnt = 0
    while lines.GetNextCell(id_list):
        point_id1 = id_list.GetId(0)
        point_id2 = id_list.GetId(1)

        # Get the coordinates of the two points
        pt0 = np.array(pts_contour.GetPoint(point_id1))
        pt1 = np.array(pts_contour.GetPoint(point_id2))

        segment_lengths[cnt] = np.linalg.norm(pt0 - pt1)
        dists_pts2surface[cnt] = abs(vtk_p2s_dist.FunctionValue((pt0 + pt1) / 2))
        cnt += 1
    return dists_pts2surface, segment_lengths


def sort_dists_and_bsizes(dists, boundary_sizes) -> Tuple[np.ndarray, np.ndarray]:
    reidx = np.argsort(np.abs(dists))
    dists_s = dists[reidx]
    boundary_sizes_s = boundary_sizes[reidx]
    return dists_s, boundary_sizes_s


def vtk_create_surface_from_polydata(
    polydata: vtk.vtkPolyData, z_offset: float
) -> vtk.vtkPolyData:
    """
    Create surfaces from multiple non-connected polylines within a vtkPolyData object.
    The function explicitly follows edges to ensure proper traversal.
    """

    z_coord = vtk_to_numpy(polydata.GetPoints().GetData())[:, -1]
    assert np.all(z_coord[0] == z_coord), "Polyline must be planar in the z-direction"

    # Ensure the input contains lines
    if polydata.GetNumberOfLines() == 0:
        raise ValueError("Input vtkPolyData must contain lines.")

    # Data structures to store results
    combined_points = vtk.vtkPoints()
    combined_cells = vtk.vtkCellArray()

    # Get points and lines from the polydata
    points = polydata.GetPoints()
    lines = polydata.GetLines()

    # Data structure to track visited lines
    visited_lines = set()

    # Traverse each polyline
    lines.InitTraversal()
    for line_id in range(lines.GetNumberOfCells()):
        if line_id in visited_lines:
            continue

        # Retrieve the current polyline
        line = vtk.vtkIdList()
        lines.GetNextCell(line)
        visited_lines.add(line_id)

        # Follow edges to construct the polyline
        polyline_points = []
        for point_id in range(line.GetNumberOfIds()):
            polyline_points.append(line.GetId(point_id))

        # Offset polyline in the Z-direction
        num_points = len(polyline_points)

        # lower and upper points
        pts_l = vtk.vtkPoints()
        pts_u = vtk.vtkPoints()
        for pid in polyline_points:
            x, y, z = points.GetPoint(pid)
            pts_l.InsertNextPoint(x, y, z - z_offset)
            pts_u.InsertNextPoint(x, y, z + z_offset)

        # Add the points from both offsets to the combined mesh
        offset = combined_points.GetNumberOfPoints()
        for i in range(num_points):
            combined_points.InsertNextPoint(pts_l.GetPoint(i))
        for i in range(num_points):
            combined_points.InsertNextPoint(pts_u.GetPoint(i))

        # Create triangles between the two polylines
        stp = 1 if num_points == 2 else 0
        for i in range(num_points - stp):
            l0 = offset + i  # Point from polyline1
            l1 = offset + (i + 1) % num_points  # Next point from polyline1
            u0 = l0 + num_points  # Corresponding point from polyline2
            u1 = l1 + num_points  # Next point from polyline2

            # Add the two triangles forming the quadrilateral
            triangle1 = vtk.vtkTriangle()
            triangle1.GetPointIds().SetId(0, l0)
            triangle1.GetPointIds().SetId(1, u0)
            triangle1.GetPointIds().SetId(2, l1)

            triangle2 = vtk.vtkTriangle()
            triangle2.GetPointIds().SetId(0, l1)
            triangle2.GetPointIds().SetId(1, u0)
            triangle2.GetPointIds().SetId(2, u1)

            combined_cells.InsertNextCell(triangle1)
            combined_cells.InsertNextCell(triangle2)

    # Finalize the combined mesh
    output_polydata = vtk.vtkPolyData()
    output_polydata.SetPoints(combined_points)
    output_polydata.SetPolys(combined_cells)

    return output_polydata


def vtk_measurements_2D(
    ref_contour: vtk.vtkPolyData,
    pred_contour: vtk.vtkPolyData,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate distances between centroids of polylines and the opposite polyline.
    Function calculates in both directions and also returns the length of line segments.

    Note: Because vtk.vtkImplicitPolyDataDistance only works for 3D vtkPolyData (i.e. meshes),
    we use a trick to create a 3D surface mesh from a polyline by sticking together two polylines with a small offset in the z-direction.

    Args:
        ref_contour (vtk.vtkPolyData): _description_
        ref_surface (vtk.vtkPolyData): _description_
        pred_contour (vtk.vtkPolyData): _description_
        pred_surface (vtk.vtkPolyData): _description_
        subdivide_iter (int, optional): _description_. Defaults to None.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: (numpy vector of distances between ref mesh vertices and pred mesh surface, numpy vector of distances between pred mesh vertices and ref mesh surface)
    """

    ref_surface = vtk_create_surface_from_polydata(ref_contour, z_offset=1)
    pred_surface = vtk_create_surface_from_polydata(pred_contour, z_offset=1)

    # fmt: off
    dists_ref2pred, segment_lengths_ref = vtk_2D_centroid2surface_dist_length(ref_contour, pred_surface)
    dists_pred2ref, segment_lengths_pred = vtk_2D_centroid2surface_dist_length(pred_contour, ref_surface)    
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


def vtk_measurements_3D(
    ref_mesh: vtk.vtkPolyData,
    pred_mesh: vtk.vtkPolyData,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate distances between centroids of triangles and surface represented by the mesh.
    Function calculates in both directions and also returns the areas of surface elements (surfels).

    Args:
        ref_mesh (vtk.vtkPolyData): Mesh created from reference segmentation
        pred_mesh (vtk.vtkPolyData): Mesh created from predicted segmentation

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: numpy vector with distances from ref to pred mesh and ref boundary sizes, and vice-versa
    """

    vtk_p2v_dist = vtk.vtkDistancePolyDataFilter()
    vtk_p2v_dist.SetInputData(0, ref_mesh)
    vtk_p2v_dist.SetInputData(1, pred_mesh)
    vtk_p2v_dist.SignedDistanceOn()
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

    # sort distances and boundary sizes
    dists_ref2pred, surfel_areas_ref = sort_dists_and_bsizes(
        dists_ref2pred, surfel_areas_ref
    )
    dists_pred2ref, surfel_areas_pred = sort_dists_and_bsizes(
        dists_pred2ref, surfel_areas_pred
    )

    return dists_ref2pred, surfel_areas_ref, dists_pred2ref, surfel_areas_pred


def vtk_voxelizer(mesh_vtk: vtk.vtkPolyData, meta_sitk: sitk.Image):
    assert isinstance(mesh_vtk, vtk.vtkPolyData), "Mesh must be vtkPolyData"
    assert isinstance(meta_sitk, sitk.Image), "Segmentation must be SimpleITK image"

    # check for empty mesh or empty image
    if mesh_vtk is None or np.prod(meta_sitk.GetSize()) == 0:
        return sitk.Image()

    vtkImage = sitk2vtk(meta_sitk)

    ndim = meta_sitk.GetDimension()
    if not np.allclose(
        meta_sitk.GetDirection(), np.eye(meta_sitk.GetDimension()).flatten()
    ):
        # vtk does not provide support for non-standard direction of the target image,
        # so we take care of this manually
        direction = np.array(meta_sitk.GetDirection()).reshape(ndim, ndim)

        T = np.eye(4)
        T[:ndim, :ndim] = direction
        T = np.linalg.inv(T)

        # Set up the transform
        vtk_affine = vtk.vtkTransform()
        vtk_affine.SetMatrix(T.flatten())
        # Apply the transform
        vtk_transform = vtk.vtkTransformPolyDataFilter()
        vtk_transform.SetInputData(mesh_vtk)
        vtk_transform.SetTransform(vtk_affine)
        vtk_transform.Update()
        mesh_vtk = vtk_transform.GetOutput()

    spacing = vtkImage.GetSpacing()
    origin = vtkImage.GetOrigin()
    extent = vtkImage.GetExtent()

    # polygonal data --> image stencil:
    poly2stenc = vtk.vtkPolyDataToImageStencil()
    poly2stenc.SetInputData(mesh_vtk)
    poly2stenc.SetOutputOrigin(origin)
    poly2stenc.SetOutputSpacing(spacing)
    poly2stenc.SetOutputWholeExtent(extent)
    poly2stenc.Update()
    stenc = poly2stenc.GetOutput()

    vtkimageStencilToImage = vtk.vtkImageStencilToImage()
    vtkimageStencilToImage.SetInputData(stenc)
    vtkimageStencilToImage.SetOutsideValue(0)
    vtkimageStencilToImage.SetInsideValue(1)
    vtkimageStencilToImage.Update()

    voxelized_sitk = vtk2sitk(vtkimageStencilToImage.GetOutput())

    # for 2D cases, vtk adds axial dim, so we remove it
    if ndim == 2:
        voxelized_sitk = voxelized_sitk[:, :, 0]
    voxelized_sitk.SetDirection(meta_sitk.GetDirection())

    return voxelized_sitk


def get_mesh_bounds(mesh: vtk.vtkPolyData) -> np.ndarray:
    bounds = np.array(mesh.GetBounds())
    if np.allclose(bounds, (1.0, -1.0, 1.0, -1.0, 1.0, -1.0)):
        bounds[:] = np.nan
    return bounds


def vtk_meshes_bbox_sitk_image(
    mesh1: vtk.vtkPolyData,
    mesh2: vtk.vtkPolyData,
    spacing: tuple,
    tolerance: tuple = None,
) -> sitk.Image:
    ndim = len(spacing)

    # create a meta image SimpleITK that encompasses both masks
    ref_b = get_mesh_bounds(mesh1)
    pred_b = get_mesh_bounds(mesh2)
    ref_origin, ref_diagonal = ref_b[::2], ref_b[1::2]
    pred_origin, pred_diagonal = pred_b[::2], pred_b[1::2]

    # find element-wise minimum and maximum
    origin = np.nanmin((ref_origin, pred_origin), axis=0)[:ndim]
    diagonal = np.nanmax((ref_diagonal, pred_diagonal), axis=0)[:ndim]

    if tolerance is not None:
        tolerance = np.array(tolerance)
        assert np.all(tolerance >= 0), "Tolerance must be positive"
        origin -= tolerance
        diagonal += tolerance

    sitk_size = np.ceil((diagonal - origin) / np.array(spacing)).astype(int)

    meta_sitk = np2sitk(np.zeros(sitk_size, dtype=np.uint8), spacing=spacing)
    meta_sitk.SetOrigin(origin)

    return meta_sitk


def vtk_create_sphere(radius: float) -> vtk.vtkPolyData:
    sphere_source = vtk.vtkSphereSource()
    sphere_source.SetRadius(radius)
    sphere_source.SetThetaResolution(64)
    sphere_source.SetPhiResolution(64)
    sphere_source.Update()
    return sphere_source.GetOutput()


def extract_cirle_from_sphere(vtk_sphere):
    # Define the cutting plane (z = 0)
    plane = vtk.vtkPlane()
    plane.SetOrigin(0.0, 0.0, 0.0)  # Origin of the plane
    plane.SetNormal(0.0, 0.0, 1.0)  # Normal vector (perpendicular to the Z-axis)

    # Use vtkCutter to slice the sphere with the plane
    cutter = vtk.vtkCutter()
    cutter.SetInputData(vtk_sphere)
    cutter.SetCutFunction(plane)  # Set the cutting plane
    cutter.Update()
    return cutter.GetOutput()


def create_synthetic_examples_3d(
    r1: float, r2: float, spacing: tuple
) -> vtk.vtkPolyData:
    # 3D
    vtk_mesh1 = vtk_create_sphere(r1)
    vtk_mesh2 = vtk_create_sphere(r2)

    # create a meta image SimpleITK that encompasses both masks
    meta_sitk = vtk_meshes_bbox_sitk_image(
        vtk_mesh1, vtk_mesh2, spacing, tolerance=5 * np.array(spacing)
    )

    sitk_mask1 = vtk_voxelizer(vtk_mesh1, meta_sitk)
    sitk_mask2 = vtk_voxelizer(vtk_mesh2, meta_sitk)

    return vtk_mesh1, vtk_mesh2, sitk_mask1, sitk_mask2


def create_synthetic_examples_2d(
    r1: float, r2: float, spacing: tuple
) -> vtk.vtkPolyData:
    # 3D
    vtk_mesh1_3d = vtk_create_sphere(r1)
    vtk_mesh2_3d = vtk_create_sphere(r2)

    # 2D
    spacing = spacing[:2]
    vtk_mesh1 = extract_cirle_from_sphere(vtk_mesh1_3d)
    vtk_mesh2 = extract_cirle_from_sphere(vtk_mesh2_3d)

    # create a meta image SimpleITK that encompasses both masks
    meta_sitk = vtk_meshes_bbox_sitk_image(
        vtk_mesh1, vtk_mesh2, spacing, tolerance=5 * np.array(spacing)
    )
    sitk_mask1 = vtk_voxelizer(vtk_mesh1, meta_sitk)
    sitk_mask2 = vtk_voxelizer(vtk_mesh2, meta_sitk)

    return vtk_mesh1, vtk_mesh2, sitk_mask1, sitk_mask2


def vtk_write_polydata(vtk_polydata: vtk.vtkPolyData, pth: Union[str, Path]):
    assert isinstance(vtk_polydata, vtk.vtkPolyData), "Unknown mesh type"

    writer = vtk.vtkOBJWriter()
    writer.SetInputData(vtk_polydata)
    writer.SetFileName(str(pth))
    writer.Write()
