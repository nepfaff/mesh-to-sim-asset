"""Utility functions for working with SDF files and converting meshes to SDF format."""

import logging

from pathlib import Path
from typing import Tuple

import coacd
import numpy as np
import trimesh

from lxml import etree as ET
from pydrake.all import (
    CalcSpatialInertia,
    SpatialInertia,
    SurfaceTriangle,
    TriangleSurfaceMesh,
)

# Set up logger.
logger = logging.getLogger(__name__)


def format_xml_for_pretty_print(element: ET.Element) -> None:
    """
    Format an XML element tree to enable proper pretty printing.

    This function adds proper indentation by setting text and tail attributes
    for all elements in the tree.

    Args:
        element: The root element to format
    """

    def indent(elem, level=0):
        i = "\n" + level * "  "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "  "
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for child in elem:
                indent(child, level + 1)
            if not child.tail or not child.tail.strip():
                child.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i

    indent(element)


def calc_mesh_spatial_inertia(
    mesh: trimesh.Trimesh, mass: float
) -> SpatialInertia | None:
    """Calculates the spatial inertia of a mesh.

    Args:
        mesh (trimesh.Trimesh): The mesh.
        mass (float): The mass in kg of the object.

    Returns:
        SpatialInertia: The spatial inertia of the mesh about the body origin and
        expressed in the body frame M_BBo_B. None if the calculation fails.
    """
    # Create a Drake mesh.
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.faces)
    triangles_drake = np.array([SurfaceTriangle(t[0], t[1], t[2]) for t in triangles])
    mesh_drake = TriangleSurfaceMesh(triangles_drake, vertices)

    # Guard against zero or negative volume meshes.
    if mesh.volume <= 0:
        logger.warning(
            f"Mesh has non-positive volume ({mesh.volume}), cannot compute inertia."
        )
        return None

    try:
        M_BBo_B: SpatialInertia = CalcSpatialInertia(
            mesh=mesh_drake, density=mass / mesh.volume
        )
    except ValueError as e:
        logger.error(f"Error calculating inertia: {e}")
        return None

    if not M_BBo_B.IsPhysicallyValid():
        logger.warning("Inertia is not physically valid.")
        return None

    return M_BBo_B


def calc_bounding_box_inertia(
    bounds: np.ndarray, mass: float
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate center of mass and inertia for a bounding box approximation.

    Uses the inertia of a solid rectangular box with the given dimensions.

    Args:
        bounds: Mesh bounds array of shape (2, 3) with [min, max] corners.
        mass: Mass in kg.

    Returns:
        Tuple of (center_of_mass, moment_of_inertia) where moment_of_inertia is diagonal.
    """
    center_of_mass = (bounds[0] + bounds[1]) / 2
    dims = bounds[1] - bounds[0]  # (w, h, d)
    w, h, d = dims[0], dims[1], dims[2]

    # Inertia of a solid box about its center of mass.
    ixx = (1.0 / 12.0) * mass * (h**2 + d**2)
    iyy = (1.0 / 12.0) * mass * (w**2 + d**2)
    izz = (1.0 / 12.0) * mass * (w**2 + h**2)

    moment_of_inertia = np.diag([ixx, iyy, izz])
    return center_of_mass, moment_of_inertia


def calc_mesh_com_and_inertia(
    mesh: trimesh.Trimesh,
    mass: float,
    frame: np.ndarray | None = None,
    use_drake: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates the mesh's center of mass and moment of inertia by assuming a uniform
    mass density.

    Args:
        mesh (trimesh.Trimesh): The mesh.
        mass (float): The mass in kg of the object.
        frame (np.ndarray, optional): The frame that the moment of inertia should be
            expressed in. If None, the center of mass is used. This is a (4,4)
            homogenous transformation matrix.
        use_drake (bool): Whether to use Drake's spatial inertia calculation instead of
            trimesh's.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple of (center_of_mass, moment_of_inertia):
            center_of_mass: The center of mass of shape (3,).
            moment_of_inertia: The moment of inertia of shape (3,3).
        Returns None, None if the inertia calculation fails.
    """
    if use_drake:
        if frame is not None:
            raise NotImplementedError("Frame is not yet supported for Drake.")

        M_BBo_B = calc_mesh_spatial_inertia(mesh=mesh, mass=mass)
        if M_BBo_B is None:
            return None, None

        center_of_mass = M_BBo_B.get_com()  # p_BoBcm_B
        # Express the inertia about the center of mass.
        M_BBcm_B = M_BBo_B.Shift(center_of_mass)
        moment_of_inertia = M_BBcm_B.CalcRotationalInertia().CopyToFullMatrix3()
    else:
        mesh.density = mass / mesh.volume
        moment_of_inertia = (
            mesh.moment_inertia if frame is None else mesh.moment_inertia_frame(frame)
        )
        center_of_mass = mesh.center_mass
    return center_of_mass, moment_of_inertia


def add_inertial_properties_element(
    mass: float, mesh: trimesh.Trimesh, link_item: ET.Element
) -> ET.Element:
    """Adds an inertial properties element for a link.

    Args:
        mass (float): The mass in kg of the mesh.
        mesh (trimesh.Trimesh): The mesh.
        link_item (ET.Element): The link element.

    Returns:
        ET.Element: The inertial properties element.
    """
    center_of_mass, moment_of_inertia = calc_mesh_com_and_inertia(mesh=mesh, mass=mass)
    inertial_item = ET.SubElement(link_item, "inertial")
    mass_item = ET.SubElement(inertial_item, "mass")
    mass_item.text = str(mass)
    if center_of_mass is None or moment_of_inertia is None:
        # Inertia calculation failed. Use bounding box inertia as fallback.
        center_of_mass, moment_of_inertia = calc_bounding_box_inertia(mesh.bounds, mass)
        logger.warning(
            f"Using bounding box inertia fallback for mesh with mass {mass:.4f} kg"
        )
    com_item = ET.SubElement(inertial_item, "pose")
    com_item.text = (
        f"{center_of_mass[0]:.5f} {center_of_mass[1]:.5f} {center_of_mass[2]:.5f} 0 0 0"
    )
    inertia_item = ET.SubElement(inertial_item, "inertia")
    for i in range(3):
        for j in range(i, 3):
            item = ET.SubElement(inertia_item, "i" + "xyz"[i] + "xyz"[j])
            # Use full precision to prevent inertia from becoming physically invalid.
            item.text = f"{moment_of_inertia[i, j]}"
    return inertial_item


def add_inertial_properties_element_multiple_meshes(
    masses: list[float], meshes: list[trimesh.Trimesh], link_item: ET.Element
) -> ET.Element:
    """Adds an inertial properties element for a link with multiple meshes.

    Args:
        masses (list[float]): The masses in kg of the meshes.
        meshes (list[trimesh.Trimesh]): The meshes that have been transformed to the
            desired pose.
        link_item (ET.Element): The link element.

    Returns:
        ET.Element: The inertial properties element.
    """
    if not len(masses) == len(meshes):
        raise ValueError("Number of masses must match number of meshes.")

    inertial_item = ET.SubElement(link_item, "inertial")
    mass_item = ET.SubElement(inertial_item, "mass")
    mass_item.text = str(sum(masses))

    inertias: list[SpatialInertia] = []
    for mass, mesh in zip(masses, meshes):
        M_BBo_B = calc_mesh_spatial_inertia(mesh=mesh, mass=mass)
        if M_BBo_B is not None:
            inertias.append(M_BBo_B)

    if len(inertias) == 0:
        # Inertia calculation failed. Use combined bounding box inertia as fallback.
        total_mass = sum(masses)
        if total_mass > 0:
            # Compute combined bounding box of all meshes.
            all_bounds = np.array([mesh.bounds for mesh in meshes])
            combined_min = np.min(all_bounds[:, 0, :], axis=0)
            combined_max = np.max(all_bounds[:, 1, :], axis=0)
            combined_bounds = np.array([combined_min, combined_max])
            center_of_mass, moment_of_inertia = calc_bounding_box_inertia(
                combined_bounds, total_mass
            )
            logger.warning(
                f"Using bounding box inertia fallback for {len(meshes)} meshes "
                f"with total mass {total_mass:.4f} kg"
            )
            com_item = ET.SubElement(inertial_item, "pose")
            com_item.text = (
                f"{center_of_mass[0]:.5f} {center_of_mass[1]:.5f} "
                f"{center_of_mass[2]:.5f} 0 0 0"
            )
            inertia_item = ET.SubElement(inertial_item, "inertia")
            for i in range(3):
                for j in range(i, 3):
                    item = ET.SubElement(inertia_item, "i" + "xyz"[i] + "xyz"[j])
                    item.text = f"{moment_of_inertia[i, j]}"
        return inertial_item

    # Sum up the inertias. All inertias have same Bo and B frames as the meshes are
    # expressed in the same frame.
    M_BBo_B: SpatialInertia = inertias[0]
    for inertia in inertias[1:]:
        M_BBo_B += inertia

    # Express the inertia about the center of mass.
    center_of_mass = M_BBo_B.get_com()
    M_BBcm_B = M_BBo_B.Shift(center_of_mass)
    moment_of_inertia = M_BBcm_B.CalcRotationalInertia().CopyToFullMatrix3()

    # Add the inertial properties.
    com_item = ET.SubElement(inertial_item, "pose")
    com_item.text = (
        f"{center_of_mass[0]:.5f} {center_of_mass[1]:.5f} {center_of_mass[2]:.5f} 0 0 0"
    )
    inertia_item = ET.SubElement(inertial_item, "inertia")
    for i in range(3):
        for j in range(i, 3):
            item = ET.SubElement(inertia_item, "i" + "xyz"[i] + "xyz"[j])
            # Use full precision to prevent inertia from becoming physically invalid.
            item.text = f"{moment_of_inertia[i, j]}"
    return inertial_item


def add_mesh_element(
    mesh_path: Path,
    link_item: ET.Element,
    scale: float = 1.0,
    is_collision: bool = False,
    name_prefix: str = "",
) -> ET.Element:
    """Adds a mesh element to an SDF file.

    Args:
        mesh_path (Path): The path to the mesh. Ideally, this is a relative path with
            respect to the output SDF file.
        link_item (ET.Element): The link element.
        scale (float): The scale factor to apply to the mesh.
        is_collision (bool): Whether the mesh is a collision mesh.
        name_prefix (str): The prefix to add to the mesh name.

    Returns:
        ET.Element: The visual or collision element.
    """
    item = (
        ET.SubElement(link_item, "collision", name=f"{name_prefix}collision")
        if is_collision
        else ET.SubElement(link_item, "visual", name=f"{name_prefix}visual")
    )
    geometry_item = ET.SubElement(item, "geometry")
    mesh_item = ET.SubElement(geometry_item, "mesh")
    uri_item = ET.SubElement(mesh_item, "uri")
    uri_item.text = mesh_path.as_posix()
    scale_item = ET.SubElement(mesh_item, "scale")
    scale_item.text = f"{scale:.3f} {scale:.3f} {scale:.3f}"
    return item


def add_compliant_proximity_properties_element(
    collision_item: ET.Element,
    hydroelastic_modulus: float,
    hunt_crossley_dissipation: float | None = None,
    mu_dynamic: float | None = None,
    mu_static: float | None = None,
) -> ET.Element:
    """Adds a compliant Hydroelasticproximity properties element to an SDF file.

    Args:
        collision_item (ET.Element): The collision element.
        hydroelastic_modulus (float): The hydroelastic modulus.
        hunt_crossley_dissipation (float | None): The hunt-crossley dissipation.
        mu_dynamic (float | None): The dynamic friction coefficient.
        mu_static (float | None): The static friction coefficient.

    Returns:
        ET.Element: The proximity properties element.
    """
    proximity_item = ET.SubElement(
        collision_item, "{drake.mit.edu}proximity_properties"
    )
    ET.SubElement(proximity_item, "{drake.mit.edu}compliant_hydroelastic")
    hydroelastic_moulus_item = ET.SubElement(
        proximity_item, "{drake.mit.edu}hydroelastic_modulus"
    )
    hydroelastic_moulus_item.text = f"{hydroelastic_modulus:.3e}"
    if hunt_crossley_dissipation is not None:
        hunt_crossley_dissipation_item = ET.SubElement(
            proximity_item, "{drake.mit.edu}hunt_crossley_dissipation"
        )
        hunt_crossley_dissipation_item.text = f"{hunt_crossley_dissipation:.3f}"
    if mu_dynamic is not None:
        mu_dynamic_item = ET.SubElement(proximity_item, "{drake.mit.edu}mu_dynamic")
        mu_dynamic_item.text = f"{mu_dynamic:.3f}"
    if mu_static is not None:
        mu_static_item = ET.SubElement(proximity_item, "{drake.mit.edu}mu_static")
        mu_static_item.text = f"{mu_static:.3f}"
    return proximity_item


def add_convex_decomposition_collision_element_with_proximity_properties(
    link_item: ET.Element,
    collision_mesh: trimesh.Trimesh,
    mesh_parts_dir_name: str,
    output_path: Path,
    hydroelastic_modulus: float | None = None,
    material_modulus: float | None = None,
    use_coacd: bool = False,
    coacd_kwargs: dict | None = None,
    vhacd_kwargs: dict | None = None,
    hunt_crossley_dissipation: float | None = None,
    mu_dynamic: float | None = None,
    mu_static: float | None = None,
    collision_name_prefix: str = "",
    resolution_values: list[float] | None = None,
    volume_change_threshold: float = 0.01,
    max_convex_hull: int = -1,
) -> None:
    """Adds a convex decomposition collision element with proximity properties to an
    SDF file.

    Args:
        link_item (ET.Element): The link element.
        collision_mesh (trimesh.Trimesh): The collision mesh.
        mesh_parts_dir_name (str): The name of the mesh parts directory.
        output_path (Path): The path to the output SDFormat file.
        hydroelastic_modulus (float | None): The Hydroelastic Modulus. The default value
            leads to low compliance. See
            https://drake.mit.edu/doxygen_cxx/group__hydroelastic__user__guide.html
            for how to pick a value. Only used if material_modulus is not provided.
        material_modulus (float | None): The material modulus that determines pressure
            as a function of depth in meters.
        use_coacd (bool): Whether to use CoACD instead of VHACD for convex decomposition.
        coacd_kwargs (dict | None): The CoACD-specific parameters.
        vhacd_kwargs (dict | None): The VHACD-specific parameters.
        hunt_crossley_dissipation (float | None): The optional Hydroelastic
            Hunt-Crossley dissipation (s/m). See
            https://drake.mit.edu/doxygen_cxx/group__hydroelastic__user__guide.html
            for how to pick a value.
        mu_dynamic (float | None): The coefficient of dynamic friction.
        mu_static (float | None): The coefficient of static friction.
        collision_name_prefix (str): The prefix to add to the collision name.
        resolution_values (list[float] | None): List of resolution values to try in order.
            For CoACD, these are threshold values (lower = higher resolution).
            If None, uses original single-pass behavior.
        volume_change_threshold (float): Maximum absolute volume improvement per additional
            part to continue refinement. Measures -(V_n - V_{n-1}) / (N_n - N_{n-1}).
            Default is 0.001 m³.
        max_convex_hull (int): Maximum number of convex hulls for CoACD decomposition.
            -1 means no limit (default). Only works when merge is enabled (default).
    """
    # Validate input.
    if hydroelastic_modulus is not None and material_modulus is not None:
        raise ValueError("Cannot use both hydroelastic_modulus and material_modulus.")

    # If neither modulus is provided, we'll use point contact (no proximity properties).
    use_hydroelastic = hydroelastic_modulus is not None or material_modulus is not None

    # Compute the convex decomposition.
    mesh_piece_paths = perform_convex_decomposition(
        mesh=collision_mesh,
        mesh_parts_dir_name=mesh_parts_dir_name,
        mesh_dir=output_path.parent,
        use_coacd=use_coacd,
        coacd_kwargs=coacd_kwargs,
        vhacd_kwargs=vhacd_kwargs,
        resolution_values=resolution_values,
        volume_change_threshold=volume_change_threshold,
        max_convex_hull=max_convex_hull,
    )

    # Delete meshes with volume less than 1 mm³ (1e-9 m³).
    meshes_to_delete = [
        path
        for path in mesh_piece_paths
        if trimesh.load_mesh(path, skip_materials=True).volume < 1e-9
    ]
    for mesh_piece_path in meshes_to_delete:
        mesh_piece_path.unlink()
    if len(meshes_to_delete) > 0:
        logger.info(
            f"Deleted {len(meshes_to_delete)} meshes with volume less than 1 mm³."
        )
    mesh_piece_paths = [
        path for path in mesh_piece_paths if path not in meshes_to_delete
    ]

    # Add the collision elements.
    for i, mesh_piece_path in enumerate(mesh_piece_paths):
        if use_hydroelastic and material_modulus is not None:
            # Compute the hydroelastic modulus by scaling the material modulus by half
            # the minimum OBB dimension.
            mesh_piece = trimesh.load_mesh(mesh_piece_path, skip_materials=True)
            min_obb_dim = float(np.min(trimesh.bounds.oriented_bounds(mesh_piece)[1]))
            hydroelastic_modulus = material_modulus * min_obb_dim / 2

        mesh_piece_path = mesh_piece_path.relative_to(output_path.parent)
        collision_item = ET.SubElement(
            link_item,
            "collision",
            name=f"{collision_name_prefix}collision_{i:03d}",
        )
        geometry_item = ET.SubElement(collision_item, "geometry")
        mesh_item = ET.SubElement(geometry_item, "mesh")
        uri_item = ET.SubElement(mesh_item, "uri")
        uri_item.text = mesh_piece_path.as_posix()
        ET.SubElement(mesh_item, "{drake.mit.edu}declare_convex")

        # Add proximity properties for hydroelastic contact.
        if use_hydroelastic:
            add_compliant_proximity_properties_element(
                collision_item=collision_item,
                hydroelastic_modulus=hydroelastic_modulus,
                hunt_crossley_dissipation=hunt_crossley_dissipation,
                mu_dynamic=mu_dynamic,
                mu_static=mu_static,
            )


def create_vtk_sdf_file(
    output_path: Path,
    visual_mesh_path: Path,
    collision_mesh_path: Path,
    mass: float,
    hydroelastic_modulus: float,
    mesh_for_physics_path: Path | None = None,
    scale: float = 1.0,
    hunt_crossley_dissipation: float | None = None,
    mu_dynamic: float | None = None,
    mu_static: float | None = None,
) -> None:
    """Creates an SDFormat file with the collision mesh as a VTK file.

    Center of mass and moment of inertia are calculated by assuming a uniform mass
    density. The model is declared as compliant Hydroelastic.

    Args:
        output_path (Path): The path to the output SDFormat file. Must end in `.sdf`.
        visual_mesh_path (Path): The path to the mesh that will be used as the visual
            geometry.
        collision_mesh_path (Path): The path to the mesh that will be used for convex
            decomposition into collision pieces. NOTE that this mesh is expected to
            align with the visual mesh. Must be a .vtk file.
        mass (float): The mass in kg of the mesh.
        hydroelastic_modulus (float): The Hydroelastic Modulus. The default value leads
            to low compliance. See
            https://drake.mit.edu/doxygen_cxx/group__hydroelastic__user__guide.html
            for how to pick a value.
        mesh_for_physics_path (Union[Path, None]): The path to the mesh that will be
            used for physics calculations. If None, the visual mesh will be used.
        scale (float): The scale factor to apply to the mesh.
        hunt_crossley_dissipation (Union[float, None]): The optional Hydroelastic
            Hunt-Crossley dissipation (s/m). See
            https://drake.mit.edu/doxygen_cxx/group__hydroelastic__user__guide.html
            for how to pick a value.
        mu_dynamic (Union[float, None]): The coefficient of dynamic friction.
        mu_static (Union[float, None]): The coefficient of static friction.
    """
    # Handle string paths.
    visual_mesh_path = Path(visual_mesh_path)
    collision_mesh_path = Path(collision_mesh_path)
    output_path = Path(output_path)

    # Validate input.
    if not output_path.suffix.lower() == ".sdf":
        raise ValueError("Output path must end in `.sdf`.")
    if not collision_mesh_path.suffix.lower() == ".vtk":
        raise ValueError("Collision mesh must be a .vtk file.")
    if not visual_mesh_path.suffix.lower() in [".obj", ".ply", ".gltf"]:
        raise ValueError("Visual mesh must be a .obj, .ply, or .gltf file.")

    # Generate the SDFormat headers.
    name = output_path.stem
    root_item = ET.Element("sdf", version="1.7", nsmap={"drake": "drake.mit.edu"})
    model_item = ET.SubElement(root_item, "model", name=name)
    link_item = ET.SubElement(model_item, "link", name=f"{name}_body_link")
    pose_item = ET.SubElement(link_item, "pose")
    pose_item.text = "0 0 0 0 0 0"

    # Add the physical properties.
    physics_mesh = trimesh.load_mesh(
        (
            mesh_for_physics_path
            if mesh_for_physics_path is not None
            else visual_mesh_path
        ),
        skip_materials=True,
    )
    add_inertial_properties_element(mass=mass, mesh=physics_mesh, link_item=link_item)

    # Add the visual mesh.
    visual_mesh_path = visual_mesh_path.relative_to(output_path.parent)
    add_mesh_element(
        mesh_path=visual_mesh_path, link_item=link_item, scale=scale, is_collision=False
    )

    # Add the collision mesh.
    collision_mesh_path = collision_mesh_path.relative_to(output_path.parent)
    collision_item = add_mesh_element(
        mesh_path=collision_mesh_path,
        link_item=link_item,
        scale=scale,
        is_collision=True,
    )

    # Add proximity properties.
    add_compliant_proximity_properties_element(
        collision_item=collision_item,
        hydroelastic_modulus=hydroelastic_modulus,
        hunt_crossley_dissipation=hunt_crossley_dissipation,
        mu_dynamic=mu_dynamic,
        mu_static=mu_static,
    )

    ET.ElementTree(root_item).write(output_path, pretty_print=True, encoding="utf-8")


def perform_convex_decomposition(
    mesh: trimesh.Trimesh,
    mesh_parts_dir_name: str,
    mesh_dir: Path,
    use_coacd: bool = False,
    coacd_kwargs: dict | None = None,
    vhacd_kwargs: dict | None = None,
    resolution_values: list[float] | None = None,
    volume_change_threshold: float = 0.001,
    max_convex_hull: int = -1,
) -> list[Path]:
    """Given a mesh, performs a convex decomposition of it with either VHACD or CoACD.
    The resulting convex parts are saved in a subfolder named `<mesh_filename>_parts`.

    Args:
        mesh (trimesh.Trimesh): The mesh to decompose.
        mesh_parts_dir_name (str): The name of the mesh parts directory.
        mesh_dir (Path): The path to the directory that the mesh is stored in. This is
            used for creating the mesh parts directory.
        use_coacd (bool): Whether to use CoACD instead of VHACD for decomposition.
        coacd_kwargs (dict | None): The CoACD-specific parameters.
        vhacd_kwargs (dict | None): The VHACD-specific parameters.
        resolution_values (list[float] | None): List of resolution values to try in order.
            For CoACD, these are threshold values (lower = higher resolution).
            If None, uses original single-pass behavior.
        volume_change_threshold (float): Maximum absolute volume improvement (m³) per additional
            part to continue refinement. Measures -(V_n - V_{n-1}) / (N_n - N_{n-1}).
            Default is 0.001 m³.
        max_convex_hull (int): Maximum number of convex hulls for CoACD decomposition.
            -1 means no limit (default). Only works when merge is enabled (default).

    Returns:
        List[Path]: The paths of the convex pieces.
    """
    # Create a subdir for the convex parts.
    out_dir = mesh_dir / mesh_parts_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check if iterative refinement is requested.
    if resolution_values is not None:
        if not use_coacd:
            raise NotImplementedError(
                "Iterative refinement with resolution_values is only supported for CoACD"
            )
        return _perform_iterative_decomposition(
            mesh,
            out_dir,
            coacd_kwargs,
            resolution_values,
            volume_change_threshold,
            max_convex_hull,
        )

    # Single-pass behavior.
    try:
        if use_coacd:
            coacd.set_log_level("error")
            coacd_mesh = coacd.Mesh(mesh.vertices, mesh.faces)
            # Prepare CoACD kwargs with max_convex_hull if specified.
            effective_kwargs = dict(coacd_kwargs or {})
            if max_convex_hull > 0:
                effective_kwargs["max_convex_hull"] = max_convex_hull
            coacd_result = coacd.run_coacd(coacd_mesh, **effective_kwargs)
            # Convert CoACD result to trimesh objects.
            convex_pieces = []
            for vertices, faces in coacd_result:
                piece = trimesh.Trimesh(vertices, faces)
                convex_pieces.append(piece)
        else:
            vhacd_settings = vhacd_kwargs or {}
            convex_pieces = mesh.convex_decomposition(**vhacd_settings)
            if not isinstance(convex_pieces, list):
                convex_pieces = [convex_pieces]
    except Exception as e:
        raise Exception(f"Failed to perform convex decomposition: {e}")

    convex_piece_paths: list[Path] = []
    piece_idx = 0
    for part in convex_pieces:
        # Skip degenerate pieces (need at least 4 vertices for a valid 3D convex hull).
        if len(part.vertices) < 4:
            continue
        piece_name = f"convex_piece_{piece_idx:03d}.obj"
        path = out_dir / piece_name
        part.export(path)
        convex_piece_paths.append(path)
        piece_idx += 1

    return convex_piece_paths


def _perform_iterative_decomposition(
    mesh: trimesh.Trimesh,
    out_dir: Path,
    coacd_kwargs: dict | None,
    resolution_values: list[float],
    volume_change_threshold: float,
    max_convex_hull: int = -1,
) -> list[Path]:
    """Performs iterative convex decomposition with increasing resolution until
    absolute volume improvement per additional part falls below threshold.

    Args:
        mesh (trimesh.Trimesh): The mesh to decompose.
        out_dir (Path): Output directory for convex pieces.
        coacd_kwargs (dict | None): CoACD parameters (threshold will be overridden).
        resolution_values (list[float]): List of threshold values to try.
        volume_change_threshold (float): Maximum absolute volume improvement (m³) per additional
            part to continue refinement. Measures -(V_n - V_{n-1}) / (N_n - N_{n-1}).
        max_convex_hull (int): Maximum number of convex hulls. -1 means no limit.

    Returns:
        List[Path]: Paths to the optimal convex pieces.
    """
    if len(resolution_values) == 0:
        raise ValueError("resolution_values must contain at least one value")

    best_pieces = None
    previous_total_volume = None
    previous_part_count = None

    # Prepare base CoACD kwargs with max_convex_hull if specified.
    base_kwargs = coacd_kwargs.copy() if coacd_kwargs else {}
    if max_convex_hull > 0:
        base_kwargs["max_convex_hull"] = max_convex_hull

    for i, threshold_value in enumerate(resolution_values):
        logger.info(f"Trying CoACD threshold: {threshold_value}")

        try:
            # Set the threshold for this iteration.
            current_kwargs = base_kwargs.copy()
            current_kwargs["threshold"] = threshold_value

            # Run CoACD directly on the original mesh.
            coacd.set_log_level("error")
            coacd_mesh = coacd.Mesh(mesh.vertices, mesh.faces)
            coacd_result = coacd.run_coacd(coacd_mesh, **current_kwargs)

            # Convert CoACD result to trimesh objects, filtering degenerate pieces.
            convex_pieces = []
            for vertices, faces in coacd_result:
                piece = trimesh.Trimesh(vertices, faces)
                # Skip degenerate pieces (< 4 vertices can't form valid 3D hull).
                if len(piece.vertices) >= 4:
                    convex_pieces.append(piece)

            # Calculate total volume of all valid pieces.
            current_total_volume = sum(piece.volume for piece in convex_pieces)
            current_part_count = len(convex_pieces)

            logger.info(
                f"Resolution {i+1}/{len(resolution_values)}: "
                f"{current_part_count} pieces, total volume: {current_total_volume:.4f}"
            )

            # Store the first result as initial best.
            if best_pieces is None:
                best_pieces = convex_pieces
                previous_total_volume = current_total_volume
                previous_part_count = current_part_count
                logger.info(f"Stored initial result with {current_part_count} pieces")
            else:
                # Calculate absolute volume improvement per additional part.
                volume_change = current_total_volume - previous_total_volume
                part_count_change = current_part_count - previous_part_count

                logger.info(
                    f"Volume change: {volume_change:.6f} m³, "
                    f"Part count increase: {part_count_change}"
                )

                if part_count_change <= 0:
                    logger.warning(
                        f"Part count did not increase ({previous_part_count} -> "
                        f"{current_part_count}). Trying next resolution value."
                    )
                    continue

                # Absolute volume improvement per additional part:
                # -(V_n - V_{n-1}) / (N_n - N_{n-1}).
                volume_improvement_per_part = -volume_change / part_count_change

                logger.info(
                    f"Volume improvement per additional part: "
                    f"{volume_improvement_per_part:.6f} m³ (threshold: "
                    f"{volume_change_threshold} m³)"
                )

                if volume_improvement_per_part <= volume_change_threshold:
                    logger.info(
                        f"Volume improvement per part {volume_improvement_per_part:.6f} m³ "
                        f"<= threshold "
                        f"{volume_change_threshold} m³. Stopping refinement and using "
                        "previous result."
                    )
                    break
                else:
                    # Significant improvement detected, update best result.
                    best_pieces = convex_pieces
                    previous_total_volume = current_total_volume
                    previous_part_count = current_part_count
                    logger.info(
                        f"Updated best result with {current_part_count} pieces due to "
                        f"volume improvement per part {volume_improvement_per_part:.6f} "
                        f"m³ > threshold {volume_change_threshold} m³"
                    )

        except Exception as e:
            logger.warning(f"Failed decomposition at threshold {threshold_value}: {e}")
            if best_pieces is None:
                raise Exception(f"All decomposition attempts failed. Last error: {e}")
            break

    # Export the best pieces to files, filtering out degenerate pieces.
    convex_piece_paths: list[Path] = []
    piece_idx = 0
    for part in best_pieces:
        # Skip degenerate pieces (need at least 4 vertices for a valid 3D convex hull).
        if len(part.vertices) < 4:
            continue
        piece_name = f"convex_piece_{piece_idx:03d}.obj"
        path = out_dir / piece_name
        part.export(path)
        convex_piece_paths.append(path)
        piece_idx += 1

    logger.info(f"Final result: {piece_idx} convex pieces saved")
    return convex_piece_paths


def create_convex_decomposition_sdf_file(
    mesh_parts_dir_name: str,
    output_path: Path,
    visual_mesh_path: Path,
    collision_mesh_path: Path,
    mass: float,
    hydroelastic_modulus: float | None = None,
    material_modulus: float | None = None,
    mesh_for_physics_path: Path | None = None,
    hunt_crossley_dissipation: float | None = None,
    mu_dynamic: float | None = None,
    mu_static: float | None = None,
    use_coacd: bool = False,
    coacd_kwargs: dict | None = None,
    vhacd_kwargs: dict | None = None,
    resolution_values: list[float] | None = None,
    volume_change_threshold: float = 0.001,
) -> None:
    """Creates an SDFormat file with the collision mesh as a convex decomposition.

    Center of mass and moment of inertia are calculated by assuming a uniform mass
    density. The model is declared as compliant Hydroelastic.

    Args:
        mesh_parts_dir_name (str): The name of the mesh parts directory.
        output_path (Path): The path to the output SDFormat file. Must end in `.sdf`.
        visual_mesh_path (Path): The path to the mesh that will be used as the visual
        geometry.
        collision_mesh_path (Path): The path to the mesh that will be used for convex
        decomposition into collision pieces. NOTE that this mesh is expected to
        align with the visual mesh.
        mass (float): The mass in kg of the mesh.
        hydroelastic_modulus (float | None): The Hydroelastic Modulus. The default value
            leads to low compliance. See
            https://drake.mit.edu/doxygen_cxx/group__hydroelastic__user__guide.html
            for how to pick a value.
        material_modulus (float | None): The material modulus that determines pressure
            as a function of depth in meters.
        material_modulus (float | None): The material modulus that determines pressure
            as a function of depth in meters.
        mesh_for_physics_path (Union[Path, None]): The path to the mesh that will be
            used for physics calculations. If None, the visual mesh will be used.
        hunt_crossley_dissipation (Union[float, None]): The optional Hydroelastic
            Hunt-Crossley dissipation (s/m). See
            https://drake.mit.edu/doxygen_cxx/group__hydroelastic__user__guide.html
            for how to pick a value.
        mu_dynamic (Union[float, None]): The coefficient of dynamic friction.
        mu_static (Union[float, None]): The coefficient of static friction.
        use_coacd (bool): Whether to use CoACD instead of VHACD for convex decomposition.
        coacd_kwargs (dict | None): The CoACD-specific parameters.
        vhacd_kwargs (dict | None): The VHACD-specific parameters.
        resolution_values (list[float] | None): List of resolution values to try in order.
            For CoACD, these are threshold values (lower = higher resolution).
            If None, uses original single-pass behavior.
        volume_change_threshold (float): Maximum absolute volume improvement (m³) per
            additional part to continue refinement.
            Measures -(V_n - V_{n-1}) / (N_n - N_{n-1}).
    """
    # Handle string paths.
    visual_mesh_path = Path(visual_mesh_path)
    collision_mesh_path = Path(collision_mesh_path)
    output_path = Path(output_path)

    # Validate input.
    if not output_path.suffix.lower() == ".sdf":
        raise ValueError("Output path must end in `.sdf`.")
    if not visual_mesh_path.suffix.lower() in [".obj", ".ply", ".gltf"]:
        raise ValueError("Visual mesh must be a .obj, .ply, or .gltf file.")
    if (use_coacd and vhacd_kwargs is not None) or (
        not use_coacd and coacd_kwargs is not None
    ):
        raise ValueError("Cannot use both CoACD and VHACD.")
    if hydroelastic_modulus is not None and material_modulus is not None:
        raise ValueError("Cannot use both hydroelastic_modulus and material_modulus.")
    if hydroelastic_modulus is None and material_modulus is None:
        raise ValueError(
            "Must specify either hydroelastic_modulus or material_modulus."
        )

    # Generate the SDFormat headers.
    name = output_path.stem
    root_item = ET.Element("sdf", version="1.7", nsmap={"drake": "drake.mit.edu"})
    model_item = ET.SubElement(root_item, "model", name=name)
    link_item = ET.SubElement(model_item, "link", name=f"{name}_body_link")
    pose_item = ET.SubElement(link_item, "pose")
    pose_item.text = "0 0 0 0 0 0"

    # Add the physical properties.
    physics_mesh = trimesh.load_mesh(
        (
            mesh_for_physics_path
            if mesh_for_physics_path is not None
            else visual_mesh_path
        ),
        skip_materials=True,
    )
    add_inertial_properties_element(mass=mass, mesh=physics_mesh, link_item=link_item)

    # Add the visual mesh.
    visual_mesh_path = visual_mesh_path.relative_to(output_path.parent)
    add_mesh_element(
        mesh_path=visual_mesh_path, link_item=link_item, scale=1.0, is_collision=False
    )

    # Compute the convex decomposition and use it as the collision geometry.
    collision_mesh = trimesh.load(
        collision_mesh_path, skip_materials=True, force="mesh"
    )
    add_convex_decomposition_collision_element_with_proximity_properties(
        link_item=link_item,
        collision_mesh=collision_mesh,
        mesh_parts_dir_name=mesh_parts_dir_name,
        output_path=output_path,
        hydroelastic_modulus=hydroelastic_modulus,
        material_modulus=material_modulus,
        use_coacd=use_coacd,
        coacd_kwargs=coacd_kwargs,
        vhacd_kwargs=vhacd_kwargs,
        hunt_crossley_dissipation=hunt_crossley_dissipation,
        mu_dynamic=mu_dynamic,
        mu_static=mu_static,
        resolution_values=resolution_values,
        volume_change_threshold=volume_change_threshold,
    )

    ET.ElementTree(root_item).write(output_path, pretty_print=True, encoding="utf-8")


def create_rigid_hydro_sdf_file(
    output_path: Path,
    visual_obj_mesh_path: Path,
    mass: float,
    mesh_for_physics_path: Path | None = None,
    scale: float = 1.0,
    hunt_crossley_dissipation: float | None = None,
    mu_dynamic: float | None = None,
    mu_static: float | None = None,
) -> None:
    """Creates an SDFormat file with the collision mesh as a rigid hydroelastic mesh.
    The visual mesh is used as the collision mesh.

    Args:
        output_path (Path): The path to the output SDFormat file. Must end in `.sdf`.
        visual_mesh_path (Path): The path to the mesh that will be used as the visual
            geometry.
        mass (float): The mass in kg of the mesh.
        mesh_for_physics_path (Union[Path, None]): The path to the mesh that will be
            used for physics calculations. If None, the visual mesh will be used.
        scale (float): The scale factor to apply to the mesh.
        hunt_crossley_dissipation (Union[float, None]): The optional Hydroelastic
            Hunt-Crossley dissipation (s/m). See
            https://drake.mit.edu/doxygen_cxx/group__hydroelastic__user__guide.html
            for how to pick a value.
        mu_dynamic (Union[float, None]): The coefficient of dynamic friction.
        mu_static (Union[float, None]): The coefficient of static friction.
    """
    # Handle string paths.
    visual_obj_mesh_path = Path(visual_obj_mesh_path)
    output_path = Path(output_path)

    # Validate input.
    if not output_path.suffix.lower() == ".sdf":
        raise ValueError("Output path must end in `.sdf`.")
    if not visual_obj_mesh_path.suffix.lower() == ".obj":
        raise ValueError("Visual mesh must be a .obj file.")

    # Generate the SDFormat headers.
    name = output_path.stem
    root_item = ET.Element("sdf", version="1.7", nsmap={"drake": "drake.mit.edu"})
    model_item = ET.SubElement(root_item, "model", name=name)
    link_item = ET.SubElement(model_item, "link", name=f"{name}_body_link")
    pose_item = ET.SubElement(link_item, "pose")
    pose_item.text = "0 0 0 0 0 0"

    # Add the physical properties.
    physics_mesh = trimesh.load_mesh(
        (
            mesh_for_physics_path
            if mesh_for_physics_path is not None
            else visual_obj_mesh_path
        ),
        skip_materials=True,
    )
    add_inertial_properties_element(mass=mass, mesh=physics_mesh, link_item=link_item)

    # Add the visual mesh.
    visual_obj_mesh_path = visual_obj_mesh_path.relative_to(output_path.parent)
    add_mesh_element(
        mesh_path=visual_obj_mesh_path,
        link_item=link_item,
        scale=scale,
        is_collision=False,
    )

    # Add the collision mesh.
    collision_item = add_mesh_element(
        mesh_path=visual_obj_mesh_path,
        link_item=link_item,
        scale=scale,
        is_collision=True,
    )

    # Add proximity properties.
    proximity_item = ET.SubElement(
        collision_item, "{drake.mit.edu}proximity_properties"
    )
    ET.SubElement(proximity_item, "{drake.mit.edu}rigid_hydroelastic")
    if hunt_crossley_dissipation is not None:
        hunt_crossley_dissipation_item = ET.SubElement(
            proximity_item, "{drake.mit.edu}hunt_crossley_dissipation"
        )
        hunt_crossley_dissipation_item.text = f"{hunt_crossley_dissipation:.3f}"
    if mu_dynamic is not None:
        mu_dynamic_item = ET.SubElement(proximity_item, "{drake.mit.edu}mu_dynamic")
        mu_dynamic_item.text = f"{mu_dynamic:.3f}"
    if mu_static is not None:
        mu_static_item = ET.SubElement(proximity_item, "{drake.mit.edu}mu_static")
        mu_static_item.text = f"{mu_static:.3f}"

    ET.ElementTree(root_item).write(output_path, pretty_print=True, encoding="utf-8")
