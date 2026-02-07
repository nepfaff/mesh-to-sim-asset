import json
import os
import subprocess
import tempfile


def _remove_gltf_extensions_required(gltf_path: str) -> bool:
    """Remove extensionsRequired from a GLTF file for Drake/VTK compatibility.

    Args:
        gltf_path: Path to the GLTF file.

    Returns:
        True if extensionsRequired was removed, False otherwise.
    """
    try:
        with open(gltf_path, "r") as f:
            data = json.load(f)
        if "extensionsRequired" in data:
            del data["extensionsRequired"]
            with open(gltf_path, "w") as f:
                json.dump(data, f, indent=2)
            return True
    except Exception:
        pass
    return False


def canonicalize_gltf(
    input_gltf_path: str,
    output_gltf_path: str,
    canonical_orientation: dict,
    scale: float = 1.0,
    placement_options: dict | None = None,
) -> None:
    """
    Canonicalize a GLTF mesh using Blender.
    This applies the canonical orientation transform in Blender's coordinate system,
    which matches the LLM's analysis coordinate system.

    Args:
        input_gltf_path: The path to the input GLTF file.
        output_gltf_path: The path to the output GLTF file.
        canonical_orientation: The canonical orientation of the mesh from the LLM in
            format {"up_axis": "z", "front_axis": "x"}, where the up_axis is the axis
            that is aligned with the world Z axis and the front_axis is the axis that
            is aligned with the world Y axis.
        scale: The scale of the mesh in range (0, 1].
        placement_options: Placement options dict, e.g. {"on_ceiling": false, ...}.
    """
    if placement_options is None:
        placement_options = {"on_object": True}

    # Create a temporary Blender script.
    blender_script = f"""
import bpy
import bmesh
import mathutils
import numpy as np

# Clear scene.
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Import GLTF.
bpy.ops.import_scene.gltf(filepath='{input_gltf_path}')

# Find root object(s) (usually an EMPTY that contains all meshes).
top_level_objects = [obj for obj in bpy.context.scene.objects if obj.parent is None]
if not top_level_objects:
    raise RuntimeError("No top-level objects found")

# Use first root object as the scene root (can extend if needed).
root_obj = next(obj for obj in top_level_objects if obj.children)

# Make it active.
bpy.context.view_layer.objects.active = root_obj

# Apply transforms (reset first to avoid cumulative transforms).
bpy.ops.object.select_all(action='DESELECT')
root_obj.select_set(True)
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

# Apply scaling.
scale_factor = {scale}
root_obj.scale = (scale_factor, scale_factor, scale_factor)
bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

# Parse canonical orientation.
up_axis = "{canonical_orientation['up_axis']}"
front_axis = "{canonical_orientation['front_axis']}"

placement_options = {placement_options}

# Axis conversion helper.
def axis_to_vector(axis_str):
    sign = -1 if axis_str.startswith('-') else 1
    base = axis_str.lstrip('-')
    if base == 'x':
        return mathutils.Vector((sign, 0, 0))
    elif base == 'y':
        return mathutils.Vector((0, sign, 0))
    elif base == 'z':
        return mathutils.Vector((0, 0, sign))

up = axis_to_vector(up_axis)
front = axis_to_vector(front_axis)

# Ensure perpendicularity: recompute front to be orthogonal to up, preserving sign.
right = front.cross(up)

if right.length < 1e-6:
    # up and front are nearly parallel — pick arbitrary orthogonal right.
    if abs(up.x) < 0.99:
        right = up.cross(mathutils.Vector((1,0,0)))
    else:
        right = up.cross(mathutils.Vector((0,1,0)))

right.normalize()
new_front = up.cross(right)
# Preserve the original front direction sign (cross product can flip it).
if new_front.dot(front) < 0:
    new_front = -new_front
    right = -right  # Also flip right to maintain right-handed system.
new_front.normalize()
front = new_front

# Build rotation matrix to align object to canonical orientation.
current_up = mathutils.Vector((0, 0, 1))
current_front = mathutils.Vector((0, 1, 0))
current_right = mathutils.Vector((1, 0, 0))

target_matrix = mathutils.Matrix((
    right,
    front,
    up
)).transposed()

current_matrix = mathutils.Matrix((
    current_right,
    current_front,
    current_up
)).transposed()

rotation_matrix = target_matrix @ current_matrix.inverted()
rotation_matrix = rotation_matrix.to_4x4()

# Apply rotation.
root_obj.matrix_world = rotation_matrix @ root_obj.matrix_world
bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)

# Compute bounding box in world coordinates (from all mesh children).
all_mesh_verts_world = []
for obj in root_obj.children_recursive:
    if obj.type == 'MESH':
        all_mesh_verts_world.extend([obj.matrix_world @ v.co for v in obj.data.vertices])

if not all_mesh_verts_world:
    raise RuntimeError("No mesh vertices found for bounding box computation.")

xs = [v.x for v in all_mesh_verts_world]
ys = [v.y for v in all_mesh_verts_world]
zs = [v.z for v in all_mesh_verts_world]

min_x, max_x = min(xs), max(xs)
min_y, max_y = min(ys), max(ys)
min_z, max_z = min(zs), max(zs)

# Placement logic.
center_x = (min_x + max_x) / 2
center_y = (min_y + max_y) / 2
center_z = (min_z + max_z) / 2

if placement_options.get("on_ceiling", False):
    # Center x/y, top at z=0 (object just below ground).
    loc_x = -center_x
    loc_y = -center_y
    loc_z = -max_z
elif placement_options.get("on_floor", False) or placement_options.get("on_object", False):
    # Center x/y, bottom at z=0 (object just above ground). Prioritize floor over wall.
    loc_x = -center_x
    loc_y = -center_y
    loc_z = -min_z
elif placement_options.get("on_wall", False):
    # Center x/z, min_y at y=0 (object just touches x-z plane).
    loc_x = -center_x
    loc_y = -min_y
    loc_z = -center_z
else:  # fallback to floor placement
    loc_x = -center_x
    loc_y = -center_y
    loc_z = -min_z

root_obj.location = mathutils.Vector((loc_x, loc_y, loc_z))
bpy.ops.object.transform_apply(location=True, rotation=False, scale=False)

# Export.
bpy.ops.export_scene.gltf(
    filepath='{output_gltf_path}',
    export_format='GLTF_SEPARATE',
    use_selection=False
)
"""

    # Write script to temporary file.
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(blender_script)
        script_path = f.name
    try:
        # Run Blender.
        cmd = [
            "env",
            "-u",
            "LD_LIBRARY_PATH",
            "blender",
            "--background",
            "--python",
            script_path,
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8", errors="replace"
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Blender canonicalization failed:\\n{result.stderr}\\n{result.stdout}"
            )
    finally:
        # Clean up script file.
        os.unlink(script_path)


def compute_canonical_rotation_matrix(canonical_orientation: dict) -> "np.ndarray":
    """Compute 3x3 rotation matrix to canonicalize orientation.

    The rotation transforms the mesh so that:
    - The specified up_axis aligns with +Z (world up)
    - The specified front_axis aligns with +Y (world front)

    Args:
        canonical_orientation: Dict with "up_axis" and "front_axis" keys,
            e.g., {"up_axis": "z", "front_axis": "-y"}.

    Returns:
        3x3 numpy rotation matrix R such that R @ object_coords = canonical_coords.
    """
    import numpy as np

    up_axis = canonical_orientation["up_axis"]
    front_axis = canonical_orientation["front_axis"]

    def axis_to_vector(axis_str: str) -> np.ndarray:
        """Convert axis string to numpy vector."""
        sign = -1 if axis_str.startswith("-") else 1
        base = axis_str.lstrip("-+").lower()
        if base == "x":
            return np.array([sign, 0.0, 0.0])
        elif base == "y":
            return np.array([0.0, sign, 0.0])
        elif base == "z":
            return np.array([0.0, 0.0, sign])
        raise ValueError(f"Invalid axis string: {axis_str}")

    up = axis_to_vector(up_axis)
    front = axis_to_vector(front_axis)

    # Compute right axis via cross product.
    right = np.cross(front, up)
    if np.linalg.norm(right) < 1e-6:
        # up and front are nearly parallel - pick arbitrary orthogonal right.
        if abs(up[0]) < 0.99:
            right = np.cross(up, np.array([1.0, 0.0, 0.0]))
        else:
            right = np.cross(up, np.array([0.0, 1.0, 0.0]))
    right = right / np.linalg.norm(right)

    # Recompute front to ensure perpendicularity, preserving original sign.
    new_front = np.cross(up, right)
    if np.dot(new_front, front) < 0:
        new_front = -new_front
        right = -right
    front = new_front / np.linalg.norm(new_front)

    # Build the object frame matrix (columns are object's axes in world coords).
    # M_obj maps canonical basis to object's current orientation.
    object_frame = np.column_stack([right, front, up])

    # We want R such that R @ object_axis = canonical_axis.
    # R @ M_obj = I => R = M_obj^-1 = M_obj.T (since orthogonal).
    rotation_matrix = object_frame.T

    return rotation_matrix


def rotate_gltf_with_blender(gltf_path: str, rotation_matrix: "np.ndarray") -> None:
    """Rotate a GLTF mesh file using Blender.

    Args:
        gltf_path: Path to the GLTF file to rotate in-place.
        rotation_matrix: 3x3 rotation matrix to apply.
    """

    # Convert rotation matrix to a flat string for embedding in script.
    r = rotation_matrix.flatten().tolist()
    r_str = ", ".join(str(x) for x in r)

    blender_script = f"""
import bpy
import mathutils

# Clear scene.
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Import GLTF.
bpy.ops.import_scene.gltf(filepath='{gltf_path}')

# Create rotation matrix from flattened values.
r_flat = [{r_str}]
rotation_matrix = mathutils.Matrix((
    (r_flat[0], r_flat[1], r_flat[2]),
    (r_flat[3], r_flat[4], r_flat[5]),
    (r_flat[6], r_flat[7], r_flat[8])
)).to_4x4()

# Apply rotation to all mesh objects.
for obj in bpy.context.scene.objects:
    if obj.type == 'MESH':
        # Apply rotation to mesh data directly.
        mesh = obj.data
        for vertex in mesh.vertices:
            v = mathutils.Vector(vertex.co)
            vertex.co = rotation_matrix @ v

# Export GLTF.
bpy.ops.export_scene.gltf(
    filepath='{gltf_path}',
    export_format='GLTF_SEPARATE',
    use_selection=False
)
"""

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(blender_script)
        script_path = f.name

    try:
        cmd = [
            "env",
            "-u",
            "LD_LIBRARY_PATH",
            "blender",
            "--background",
            "--python",
            script_path,
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8", errors="replace"
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Blender GLTF rotation failed:\n{result.stderr}\n{result.stdout}"
            )
    finally:
        os.unlink(script_path)

    # Remove extensionsRequired for Drake/VTK compatibility.
    _remove_gltf_extensions_required(gltf_path)


def rotate_obj_mesh(obj_path: str, rotation_matrix: "np.ndarray") -> None:
    """Rotate an OBJ mesh file using trimesh.

    Args:
        obj_path: Path to the OBJ file to rotate in-place.
        rotation_matrix: 3x3 rotation matrix to apply.
    """
    import numpy as np
    import trimesh

    mesh = trimesh.load(obj_path, force="mesh")

    # Create 4x4 transform matrix.
    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix

    mesh.apply_transform(transform)
    mesh.export(obj_path)


def rotate_vtk_mesh(vtk_path: str, rotation_matrix: "np.ndarray") -> None:
    """Rotate a VTK mesh file.

    Args:
        vtk_path: Path to the VTK file to rotate in-place.
        rotation_matrix: 3x3 rotation matrix to apply.
    """
    import meshio

    mesh = meshio.read(vtk_path)
    mesh.points = (rotation_matrix @ mesh.points.T).T
    meshio.write(vtk_path, mesh)


def canonicalize_articulated_sdf(
    sdf_path: str,
    canonical_orientation: dict,
    placement_options: dict | None = None,
) -> None:
    """Canonicalize an articulated SDF asset in-place.

    This function:
    1. Computes the canonical rotation from the VLM-determined orientation
    2. Rotates all mesh files (GLTF, OBJ, VTK) about the world origin
    3. Rotates joint pose positions (NOT orientations) and joint axis vectors
    4. Rotates inertial pose positions (NOT orientations)
    5. Applies placement translation to meshes, joint poses, and inertial poses

    Key insight: We rotate joint positions and axes, but NOT joint orientations.
    If we rotated joint orientation, the axis (in joint frame) would flip sign in
    world space, which breaks joint limits - positive rotation would become the
    opposite direction. Instead, we keep joint frame orientation unchanged and
    rotate the axis vector explicitly.

    Args:
        sdf_path: Path to the SDF file.
        canonical_orientation: Dict with "up_axis" and "front_axis" keys.
        placement_options: Placement options dict (on_floor, on_ceiling, on_wall).
    """
    import logging

    import numpy as np
    import trimesh

    from lxml import etree as ET
    from scipy.spatial.transform import Rotation

    logger = logging.getLogger(__name__)

    if placement_options is None:
        placement_options = {"on_object": True}

    sdf_path = os.path.abspath(sdf_path)
    sdf_dir = os.path.dirname(sdf_path)

    # Compute the canonical rotation matrix.
    R = compute_canonical_rotation_matrix(canonical_orientation)
    scipy_rot = Rotation.from_matrix(R)

    # Check if rotation is essentially identity (no rotation needed).
    angle = np.linalg.norm(scipy_rot.as_rotvec())
    needs_rotation = angle >= 1e-6

    if needs_rotation:
        logger.info(f"Applying canonical rotation (angle={np.degrees(angle):.1f}°)")
    else:
        logger.info("Canonical orientation is already aligned, skipping rotation.")

    # Parse the SDF file.
    tree = ET.parse(sdf_path)
    root = tree.getroot()

    # Collect all mesh URIs referenced in the SDF.
    mesh_uris = set()
    for uri_elem in root.iter("uri"):
        if uri_elem.text:
            mesh_uris.add(uri_elem.text)

    # Rotate all mesh files if needed.
    if needs_rotation:
        for mesh_uri in mesh_uris:
            mesh_path = os.path.join(sdf_dir, mesh_uri)
            if not os.path.exists(mesh_path):
                logger.warning(f"Mesh file not found: {mesh_path}")
                continue

            suffix = os.path.splitext(mesh_path)[1].lower()
            try:
                if suffix in [".gltf", ".glb"]:
                    rotate_gltf_with_blender(mesh_path, R)
                    logger.info(f"Rotated GLTF: {mesh_uri}")
                elif suffix == ".obj":
                    rotate_obj_mesh(mesh_path, R)
                    logger.info(f"Rotated OBJ: {mesh_uri}")
                elif suffix == ".vtk":
                    rotate_vtk_mesh(mesh_path, R)
                    logger.info(f"Rotated VTK: {mesh_uri}")
            except Exception as e:
                logger.error(f"Failed to rotate mesh {mesh_uri}: {e}")

    # Helper to rotate a joint axis element.
    def rotate_axis_element(xyz_elem: ET.Element) -> None:
        """Rotate a <xyz> joint axis element."""
        if xyz_elem is None or xyz_elem.text is None:
            return

        xyz = np.array([float(x) for x in xyz_elem.text.strip().split()])
        new_xyz = R @ xyz
        # Normalize to handle any floating point drift.
        new_xyz = new_xyz / np.linalg.norm(new_xyz)
        xyz_elem.text = f"{new_xyz[0]:.8g} {new_xyz[1]:.8g} {new_xyz[2]:.8g}"

    def rotate_pose_position_only(pose_elem: ET.Element, rotation: np.ndarray) -> None:
        """Rotate only the position component of a pose element.

        The orientation is NOT rotated. This is used for joint poses where we
        want to keep the joint frame orientation unchanged (so joint limits
        continue to work correctly) while rotating the axis explicitly.

        Args:
            pose_elem: The <pose> XML element to transform.
            rotation: 3x3 rotation matrix to apply to position only.
        """
        if pose_elem is None or pose_elem.text is None:
            return

        values = [float(x) for x in pose_elem.text.strip().split()]
        if len(values) < 6:
            return

        # Rotate position only, keep orientation unchanged.
        pos = np.array(values[:3])
        new_pos = rotation @ pos

        pose_elem.text = (
            f"{new_pos[0]:.8g} {new_pos[1]:.8g} {new_pos[2]:.8g} "
            f"{values[3]:.8g} {values[4]:.8g} {values[5]:.8g}"
        )

    # Update joints if rotation was applied.
    # Key insight: We rotate joint position and axis, but NOT joint orientation.
    # Rotating joint orientation would flip the axis sign in world space, which
    # breaks joint limits (positive rotation would become negative direction).
    if needs_rotation:
        for joint in root.iter("joint"):
            # Rotate joint pose position (but NOT orientation).
            joint_pose = joint.find("pose")
            if joint_pose is not None:
                rotate_pose_position_only(joint_pose, R)

            # Always rotate axis vectors explicitly.
            # The axis is in joint frame, but since we're NOT rotating the joint
            # frame orientation, we must rotate the axis to get correct world dir.
            axis = joint.find("axis")
            if axis is not None:
                xyz = axis.find("xyz")
                rotate_axis_element(xyz)
            axis2 = joint.find("axis2")
            if axis2 is not None:
                xyz2 = axis2.find("xyz")
                rotate_axis_element(xyz2)

        # Rotate inertial pose positions only (not orientations).
        # Inertia tensor is relative to principal axes, which are defined by the
        # inertial pose orientation. We keep that unchanged.
        for link in root.iter("link"):
            inertial = link.find("inertial")
            if inertial is not None:
                inertial_pose = inertial.find("pose")
                if inertial_pose is not None:
                    rotate_pose_position_only(inertial_pose, R)

    # Compute combined bounding box from all meshes for placement.
    # Meshes are already rotated, so we get world-space bounds directly.
    all_verts = []
    for mesh_uri in mesh_uris:
        mesh_path = os.path.join(sdf_dir, mesh_uri)
        if not os.path.exists(mesh_path):
            continue
        suffix = os.path.splitext(mesh_path)[1].lower()
        try:
            if suffix in [".gltf", ".glb"]:
                # For GLTF, use Blender to get accurate bounds (avoids Y-up issues).
                bounds = _get_gltf_bounds_with_blender(mesh_path)
                if bounds is not None:
                    min_pt, max_pt = bounds
                    # Add corner points to approximate bounding box.
                    all_verts.append(min_pt.tolist())
                    all_verts.append(max_pt.tolist())
            else:
                mesh = trimesh.load(mesh_path, force="mesh")
                all_verts.extend(mesh.vertices.tolist())
        except Exception:
            pass

    if all_verts:
        verts = np.array(all_verts)
        min_xyz = verts.min(axis=0)
        max_xyz = verts.max(axis=0)

        center_x = (min_xyz[0] + max_xyz[0]) / 2
        center_y = (min_xyz[1] + max_xyz[1]) / 2

        # Compute placement offset. Priority: ceiling > floor/on_object > wall.
        if placement_options.get("on_ceiling", False):
            offset = np.array([-center_x, -center_y, -max_xyz[2]])
        elif placement_options.get("on_floor", False) or placement_options.get(
            "on_object", False
        ):
            # Floor/on_object: center x/y, bottom at z=0.
            offset = np.array([-center_x, -center_y, -min_xyz[2]])
        elif placement_options.get("on_wall", False):
            offset = np.array([-center_x, -min_xyz[1], -(min_xyz[2] + max_xyz[2]) / 2])
        else:  # fallback to floor
            offset = np.array([-center_x, -center_y, -min_xyz[2]])

        logger.info(
            f"Placement offset: [{offset[0]:.4f}, {offset[1]:.4f}, {offset[2]:.4f}]"
        )

        def translate_pose_element(pose_elem: ET.Element, offset: np.ndarray) -> None:
            """Translate a pose element's position by offset."""
            if pose_elem is None or pose_elem.text is None:
                return

            values = [float(x) for x in pose_elem.text.strip().split()]
            if len(values) < 6:
                return

            # Translate position only.
            pos = np.array(values[:3])
            new_pos = pos + offset

            pose_elem.text = (
                f"{new_pos[0]:.8g} {new_pos[1]:.8g} {new_pos[2]:.8g} "
                f"{values[3]:.8g} {values[4]:.8g} {values[5]:.8g}"
            )

        # Apply offset to all meshes.
        for mesh_uri in mesh_uris:
            mesh_path = os.path.join(sdf_dir, mesh_uri)
            if not os.path.exists(mesh_path):
                continue
            suffix = os.path.splitext(mesh_path)[1].lower()
            try:
                if suffix in [".gltf", ".glb"]:
                    _translate_gltf_with_blender(mesh_path, offset)
                    logger.info(f"Translated GLTF: {mesh_uri}")
                elif suffix == ".obj":
                    mesh = trimesh.load(mesh_path, force="mesh")
                    mesh.vertices += offset
                    mesh.export(mesh_path)
                    logger.info(f"Translated OBJ: {mesh_uri}")
                elif suffix == ".vtk":
                    import meshio

                    m = meshio.read(mesh_path)
                    m.points += offset
                    meshio.write(mesh_path, m)
                    logger.info(f"Translated VTK: {mesh_uri}")
            except Exception as e:
                logger.warning(f"Failed to apply placement offset to {mesh_uri}: {e}")

        # Apply offset to joint poses.
        for joint in root.iter("joint"):
            joint_pose = joint.find("pose")
            if joint_pose is not None:
                translate_pose_element(joint_pose, offset)

        # Apply offset to inertial poses.
        for link in root.iter("link"):
            inertial = link.find("inertial")
            if inertial is not None:
                inertial_pose = inertial.find("pose")
                if inertial_pose is not None:
                    translate_pose_element(inertial_pose, offset)

    # Write updated SDF (for joint axis changes).
    tree.write(sdf_path, pretty_print=True, encoding="utf-8")
    logger.info(f"Canonicalized articulated SDF: {sdf_path}")


def _get_gltf_bounds_with_blender(
    gltf_path: str,
) -> "tuple[np.ndarray, np.ndarray]|None":
    """Get bounding box of a GLTF file using Blender.

    Args:
        gltf_path: Path to the GLTF file.

    Returns:
        Tuple of (min_point, max_point) as numpy arrays, or None if failed.
    """
    import json

    import numpy as np

    blender_script = f"""
import bpy
import json

# Clear scene.
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Import GLTF.
bpy.ops.import_scene.gltf(filepath='{gltf_path}')

# Compute bounding box from all mesh objects.
min_x = min_y = min_z = float('inf')
max_x = max_y = max_z = float('-inf')

for obj in bpy.context.scene.objects:
    if obj.type == 'MESH':
        # Get vertices in world coordinates.
        for vertex in obj.data.vertices:
            world_co = obj.matrix_world @ vertex.co
            min_x = min(min_x, world_co.x)
            min_y = min(min_y, world_co.y)
            min_z = min(min_z, world_co.z)
            max_x = max(max_x, world_co.x)
            max_y = max(max_y, world_co.y)
            max_z = max(max_z, world_co.z)

# Output bounds as JSON to stdout.
if min_x != float('inf'):
    print("BOUNDS_START")
    print(json.dumps({{"min": [min_x, min_y, min_z], "max": [max_x, max_y, max_z]}}))
    print("BOUNDS_END")
"""

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(blender_script)
        script_path = f.name

    try:
        cmd = [
            "env",
            "-u",
            "LD_LIBRARY_PATH",
            "blender",
            "--background",
            "--python",
            script_path,
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8", errors="replace"
        )
        if result.returncode != 0:
            return None

        # Parse bounds from stdout.
        stdout = result.stdout
        if "BOUNDS_START" in stdout and "BOUNDS_END" in stdout:
            start = stdout.index("BOUNDS_START") + len("BOUNDS_START")
            end = stdout.index("BOUNDS_END")
            bounds_json = stdout[start:end].strip()
            bounds = json.loads(bounds_json)
            return (np.array(bounds["min"]), np.array(bounds["max"]))
        return None
    except Exception:
        return None
    finally:
        os.unlink(script_path)


def _translate_gltf_with_blender(gltf_path: str, offset: "np.ndarray") -> None:
    """Translate a GLTF mesh file using Blender.

    Args:
        gltf_path: Path to the GLTF file to translate in-place.
        offset: 3D translation offset.
    """
    offset_str = ", ".join(str(x) for x in offset.tolist())

    blender_script = f"""
import bpy
import mathutils

# Clear scene.
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Import GLTF.
bpy.ops.import_scene.gltf(filepath='{gltf_path}')

# Create offset vector.
offset = mathutils.Vector(({offset_str}))

# Apply translation to all mesh objects.
for obj in bpy.context.scene.objects:
    if obj.type == 'MESH':
        mesh = obj.data
        for vertex in mesh.vertices:
            vertex.co += offset

# Export GLTF.
bpy.ops.export_scene.gltf(
    filepath='{gltf_path}',
    export_format='GLTF_SEPARATE',
    use_selection=False
)
"""

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(blender_script)
        script_path = f.name

    try:
        cmd = [
            "env",
            "-u",
            "LD_LIBRARY_PATH",
            "blender",
            "--background",
            "--python",
            script_path,
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8", errors="replace"
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Blender GLTF translation failed:\n{result.stderr}\n{result.stdout}"
            )
    finally:
        os.unlink(script_path)

    # Remove extensionsRequired for Drake/VTK compatibility.
    _remove_gltf_extensions_required(gltf_path)
