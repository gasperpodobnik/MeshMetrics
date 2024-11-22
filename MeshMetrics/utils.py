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
        return vtk.vtkPolyData()
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
        id_last = new_points.InsertNextPoint(points.GetPoint(id_list.GetId(id_list.GetNumberOfIds()-1)))
        new_lines.InsertNextCell(2)
        new_lines.InsertCellPoint(id_mid)
        new_lines.InsertCellPoint(id_last)

    # Create a new polyline with subdivided segments
    subdivided_polyline = vtk.vtkPolyData()
    subdivided_polyline.SetPoints(new_points)
    subdivided_polyline.SetLines(new_lines)
    return subdivided_polyline

def vtkLinearSubdividePolylineToLength(polydata, max_segment_length):
    # Extract points and lines from the input polyline
    og_points = polydata.GetPoints()
    og_lines = polydata.GetLines()
    
    # Create new points and lines for the subdivided polyline
    new_points = vtk.vtkPoints()
    new_lines = vtk.vtkCellArray()
    
    og_lines.InitTraversal()
    id_list = vtk.vtkIdList()

    # Traverse each polyline segment
    while og_lines.GetNextCell(id_list):
        og_previous_point_id = None
        for i in range(id_list.GetNumberOfIds()):
            og_point_id = id_list.GetId(i)
            current_point = np.array(og_points.GetPoint(og_point_id))

            if og_previous_point_id is not None:
                # Calculate distance and direction to previous point
                previous_point = np.array(og_points.GetPoint(og_previous_point_id))
                segment_vector = current_point - previous_point
                segment_length = np.linalg.norm(segment_vector)
                
                # Determine the number of subdivisions needed
                if segment_length > max_segment_length:
                    num_subdivisions = int(np.ceil(segment_length / max_segment_length))
                    for j in range(1, num_subdivisions):
                        interpolated_point = previous_point + (j / num_subdivisions) * segment_vector
                        new_point_id = new_points.InsertNextPoint(interpolated_point)
                        new_lines.InsertNextCell(2)
                        new_lines.InsertCellPoint(new_previous_point_id)
                        new_lines.InsertCellPoint(new_point_id)
                        new_previous_point_id = new_point_id
                
                # Add the last point of the current segment
                new_point_id = new_points.InsertNextPoint(current_point)
                new_lines.InsertNextCell(2)
                new_lines.InsertCellPoint(new_previous_point_id)
                new_lines.InsertCellPoint(new_point_id)
            else:
                # Add the first point of the polyline
                new_point_id = new_points.InsertNextPoint(current_point)
            
            og_previous_point_id = og_point_id
            new_previous_point_id = new_point_id

    # Create new polydata with subdivided points and lines
    subdivided_polydata = vtk.vtkPolyData()
    subdivided_polydata.SetPoints(new_points)
    subdivided_polydata.SetLines(new_lines)
    
    return subdivided_polydata

def vtk_2D_centroid2surface_dist_length(
    pts_contour: vtk.vtkPolyData,
    surface_mesh: vtk.vtkPolyData,
) -> Tuple[np.ndarray, np.ndarray]:

    lines = pts_contour.GetLines()
    lines.InitTraversal()
    id_list = vtk.vtkIdList()
    
    assert lines.GetMaxCellSize() == 2, "This function supports segment lines that have 2 points"
    
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
        dists_pts2surface[cnt] = abs(vtk_p2s_dist.FunctionValue((pt0+pt1)/2))
        cnt += 1
    return dists_pts2surface, segment_lengths


def sort_dists_and_bsizes(dists, boundary_sizes) -> Tuple[np.ndarray, np.ndarray]:
    _sorted = np.array(sorted(zip(dists, boundary_sizes)))
    dists_s, boundary_sizes_s = _sorted[:, 0], _sorted[:, 1]
    return dists_s, boundary_sizes_s

def vtk_create_surface_from_polydata(polydata: vtk.vtkPolyData, z_offset: float) -> vtk.vtkPolyData:
    """
    Create surfaces from multiple non-connected polylines within a vtkPolyData object.
    The function explicitly follows edges to ensure proper traversal.
    """

    z_coord = vtk_to_numpy(polydata.GetPoints().GetData())[:,-1]
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
        for i in range(num_points-stp):
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

def vtk_centroids2contour_measurements(
    ref_contour: vtk.vtkPolyData,
    pred_contour: vtk.vtkPolyData,
    subdivide_iter: int = 0,
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

    for _ in range(subdivide_iter):
        ref_contour = vtkLinearSubdividePolyline(ref_contour)
        pred_contour = vtkLinearSubdividePolyline(pred_contour)        
    
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

    for _ in range(subdivide_iter):
        ref_mesh = vtk_subdivide_mesh(ref_mesh)
        pred_mesh = vtk_subdivide_mesh(pred_mesh)

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
        
    ref_dist_field = implicit_signed_distance(ref_mesh, _origin, _diagonal, _spacing)
    pred_dist_field = implicit_signed_distance(pred_mesh, _origin, _diagonal, _spacing)

    # get hollowed masks
    sitk_and = sitk.AndImageFilter()
    ref_hollowed_seg = sitk_and.Execute(ref_dist_field < tau, ref_dist_field >= 0)
    pred_hollowed_seg = sitk_and.Execute(pred_dist_field < tau, pred_dist_field >= 0)
    
    return sitk2np(ref_hollowed_seg) > 0, sitk2np(pred_hollowed_seg) > 0

def get_hollow_mask(ref_mask: sitk.Image, pred_mask: sitk.Image, tau: float) -> Tuple[np.ndarray, np.ndarray]:
    assert tau > 0, "Distance must be positive"

    edt_sitk = sitk.SignedMaurerDistanceMapImageFilter()
    edt_sitk.SquaredDistanceOff()
    edt_sitk.SetInsideIsPositive(True)
    edt_sitk.SetUseImageSpacing(True)
    ref_dist_field = edt_sitk.Execute(ref_mask > 0)
    pred_dist_field = edt_sitk.Execute(pred_mask > 0)

    # get hollowed masks
    sitk_and = sitk.AndImageFilter()
    ref_hollowed_seg = sitk_and.Execute(ref_dist_field < tau, ref_dist_field >= 0)
    pred_hollowed_seg = sitk_and.Execute(pred_dist_field < tau, pred_dist_field >= 0)
    
    return sitk2np(ref_hollowed_seg) > 0, sitk2np(pred_hollowed_seg) > 0

def vtk_voxelizer(mesh_vtk: vtk.vtkPolyData, meta_sitk: sitk.Image):
    assert isinstance(mesh_vtk, vtk.vtkPolyData), "Mesh must be vtkPolyData"
    assert isinstance(meta_sitk, sitk.Image), "Segmentation must be SimpleITK image"
    
    # check for empty mesh or empty image
    if mesh_vtk is None or np.prod(meta_sitk.GetSize()) == 0:
        return sitk.Image()
    
    vtkImage = sitk2vtk(meta_sitk)
    
    ndim = meta_sitk.GetDimension()
    if not np.allclose(meta_sitk.GetDirection(), np.eye(meta_sitk.GetDimension()).flatten()):
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
        voxelized_sitk = voxelized_sitk[:,:,0]
    voxelized_sitk.SetDirection(meta_sitk.GetDirection())

    return voxelized_sitk

def get_mesh_bounds(mesh: vtk.vtkPolyData) -> np.ndarray:
    bounds = np.array(mesh.GetBounds())
    if np.allclose(bounds, (1.0, -1.0, 1.0, -1.0, 1.0, -1.0)):
        bounds[:] = np.nan
    return bounds

def vtk_meshes_bbox_sitk_image(mesh1: vtk.vtkPolyData, mesh2: vtk.vtkPolyData, spacing: tuple, tolerance: tuple=None) -> sitk.Image:    
    ndim = len(spacing)
    
    # create a meta image SimpleITK that encompasses both masks
    ref_b = get_mesh_bounds(mesh1)
    pred_b = get_mesh_bounds(mesh2)
    ref_origin, ref_diagonal =ref_b[::2], ref_b[1::2]
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

def create_synthetic_examples_3d(r1: float, r2: float, spacing: tuple) -> vtk.vtkPolyData:
    # 3D
    vtk_mesh1 = vtk_create_sphere(r1)
    vtk_mesh2 = vtk_create_sphere(r2)
    
    # create a meta image SimpleITK that encompasses both masks
    meta_sitk = vtk_meshes_bbox_sitk_image(vtk_mesh1, vtk_mesh2, spacing, tolerance=5*np.array(spacing))

    sitk_mask1 = vtk_voxelizer(vtk_mesh1, meta_sitk)
    sitk_mask2 = vtk_voxelizer(vtk_mesh2, meta_sitk) 
    
    return vtk_mesh1, vtk_mesh2, sitk_mask1, sitk_mask2


def create_synthetic_examples_2d(r1: float, r2: float, spacing: tuple) -> vtk.vtkPolyData:
    # 3D
    vtk_mesh1_3d = vtk_create_sphere(r1)
    vtk_mesh2_3d = vtk_create_sphere(r2)
    
    # 2D
    spacing = spacing[:2]
    vtk_mesh1 = extract_cirle_from_sphere(vtk_mesh1_3d)
    vtk_mesh2 = extract_cirle_from_sphere(vtk_mesh2_3d)
    
    # create a meta image SimpleITK that encompasses both masks
    meta_sitk = vtk_meshes_bbox_sitk_image(vtk_mesh1, vtk_mesh2, spacing, tolerance=5*np.array(spacing))
    sitk_mask1 = vtk_voxelizer(vtk_mesh1, meta_sitk)
    sitk_mask2 = vtk_voxelizer(vtk_mesh2, meta_sitk)
    
    return vtk_mesh1, vtk_mesh2, sitk_mask1, sitk_mask2

def calculate_metrics_and_print_scores(dist_metrics_class, tau: float = 2):
    """Simple function to calculate metric scores and print results."""
    # Hausdorff Distance (HD), by default, HD percentile is set to 100 (equivalent to HD)
    hd100 = dist_metrics_class.hd()
    # 95th percentile HD
    hd95 = dist_metrics_class.hd(percentile=95)
    # Mean Average Surface Distance (MASD)
    masd = dist_metrics_class.masd()
    # Average Symmetric Surface Distance (ASSD)
    assd = dist_metrics_class.assd()
    # Normalized Surface Distance (NSD) with tau
    nsd2 = dist_metrics_class.nsd(tau=tau)
    # Boundary Intersection over Union (BIoU) with tau
    biou2 = dist_metrics_class.biou(tau=tau)
    
    # print results
    print(f"HD100: {hd100:.2f} mm, HD (perc=95): {hd95:.2f} mm, MASD: {masd:.2f} mm, ASSD: {assd:.2f} mm, NSD (tau={tau} mm): {nsd2*100:.2f} %, BIoU (tau={tau} mm): {biou2*100:.2f} %")