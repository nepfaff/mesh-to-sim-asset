import json
import os
import subprocess
import tempfile

from pathlib import Path

import numpy as np

from scipy.spatial.transform import Rotation


def remove_gltf_extensions_required(gltf_path: Path) -> bool:
    """Remove extensionsRequired from a GLTF file for Drake/VTK compatibility.

    VTK (used by Drake) doesn't support some GLTF extensions like
    KHR_texture_transform. When marked as required, VTK fails to load the file.

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


def convert_obj_to_gltf(input_path: Path, output_path: Path) -> Path:
    """Convert OBJ file to GLTF format using Blender.

    This ensures the model is in a right-handed coordinate system with:
    - +Y as up
    - +X as right
    - -Z as forward

    Args:
        input_path (str): Path to the input OBJ file.
        output_path (str): Path to save the output GLTF file.

    Returns:
        str: Path to the converted GLTF file.
    """
    # Use Blender to convert with correct coordinate system.
    blender_script = f"""
import bpy
import math

# Clear existing objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Import OBJ
bpy.ops.wm.obj_import(
        filepath='{input_path.as_posix()}',
        forward_axis='Y',  # +Y forward
        up_axis='Z'        # +Z up
    )

# Select all objects
bpy.ops.object.select_all(action='SELECT')

# Export as glTF
bpy.ops.export_scene.gltf(
    filepath='{output_path.as_posix()}',
    export_format='GLTF_SEPARATE',
    use_selection=True,
    export_yup=True
)
"""

    # Write to a *non-deleted* temporary file.
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(blender_script)
        script_path = f.name

    try:
        # Run Blender.
        result = subprocess.run(
            [
                "env",
                "-u",
                "LD_LIBRARY_PATH",
                "blender",
                "--background",
                "--python",
                script_path,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Blender conversion failed:\n{result.stderr}\n{result.stdout}"
            )
    finally:
        # Clean up temporary script.
        os.remove(script_path)

    return output_path


def convert_ply_to_gltf(input_path: Path, output_path: Path) -> Path:
    """Convert PLY file to GLTF format using Blender.

    This ensures the model is in a right-handed coordinate system with:
    - +Y as up
    - +X as right
    - -Z as forward

    Args:
        input_path (Path): Path to the input PLY file.
        output_path (Path): Path to save the output GLTF file.

    Returns:
        Path: Path to the converted GLTF file.
    """
    # Use Blender to convert with correct coordinate system.
    blender_script = f"""
import bpy
import math

# Clear existing objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Import PLY
bpy.ops.wm.ply_import(
    filepath='{input_path.as_posix()}'
)

# Select all objects
bpy.ops.object.select_all(action='SELECT')

# Export as glTF
bpy.ops.export_scene.gltf(
    filepath='{output_path.as_posix()}',
    export_format='GLTF_SEPARATE',
    use_selection=True,
    export_yup=True
)
"""

    # Write to a *non-deleted* temporary file.
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(blender_script)
        script_path = f.name

    try:
        # Run Blender.
        result = subprocess.run(
            [
                "env",
                "-u",
                "LD_LIBRARY_PATH",
                "blender",
                "--background",
                "--python",
                script_path,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Blender conversion failed:\n{result.stderr}\n{result.stdout}"
            )
    finally:
        # Clean up temporary script.
        os.remove(script_path)

    return output_path


def convert_fbx_to_gltf(input_path: Path, output_path: Path) -> Path:
    """Convert FBX file to GLTF format using Blender.

    This ensures the model is in a right-handed coordinate system with:
    - +Y as up
    - +X as right
    - -Z as forward

    Args:
        input_path (Path): Path to the input FBX file.
        output_path (Path): Path to save the output GLTF file.

    Returns:
        Path: Path to the converted GLTF file.
    """
    # Use Blender to convert with correct coordinate system.
    blender_script = f"""
import bpy
import math

# Clear existing objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Import FBX
bpy.ops.import_scene.fbx(
    filepath='{input_path.as_posix()}'
)

# Select all objects
bpy.ops.object.select_all(action='SELECT')

# Export as glTF
bpy.ops.export_scene.gltf(
    filepath='{output_path.as_posix()}',
    export_format='GLTF_SEPARATE',
    use_selection=True,
    export_yup=True
)
"""

    # Write to a *non-deleted* temporary file.
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(blender_script)
        script_path = f.name

    try:
        # Run Blender.
        result = subprocess.run(
            [
                "env",
                "-u",
                "LD_LIBRARY_PATH",
                "blender",
                "--background",
                "--python",
                script_path,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Blender conversion failed:\n{result.stderr}\n{result.stdout}"
            )
    finally:
        # Clean up temporary script.
        os.remove(script_path)

    return output_path


def convert_blend_to_gltf(input_path: Path, output_path: Path) -> Path:
    """Convert Blender file to GLTF format using Blender.

    This ensures the model is in a right-handed coordinate system with:
    - +Y as up
    - +X as right
    - -Z as forward

    Args:
        input_path (Path): Path to the input Blender file.
        output_path (Path): Path to save the output GLTF file.

    Returns:
        Path: Path to the converted GLTF file.
    """
    # Use Blender to convert with correct coordinate system.
    blender_script = f"""
import bpy

# Open the Blender file
bpy.ops.wm.open_mainfile(filepath='{input_path.as_posix()}')

# Select all objects
bpy.ops.object.select_all(action='SELECT')

# Export as glTF
bpy.ops.export_scene.gltf(
    filepath='{output_path.as_posix()}',
    export_format='GLTF_SEPARATE',
    use_selection=True,
    export_yup=True
)
"""

    # Write to a *non-deleted* temporary file.
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(blender_script)
        script_path = f.name

    try:
        # Run Blender.
        result = subprocess.run(
            [
                "env",
                "-u",
                "LD_LIBRARY_PATH",
                "blender",
                "--background",
                "--python",
                script_path,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Blender conversion failed:\n{result.stderr}\n{result.stdout}"
            )
    finally:
        # Clean up temporary script.
        os.remove(script_path)

    return output_path


def convert_dae_to_gltf(input_path: Path, output_path: Path) -> Path:
    """Convert DAE (COLLADA) file to GLTF format using Blender.

    This ensures the model is in a right-handed coordinate system with:
    - +Y as up
    - +X as right
    - -Z as forward

    Args:
        input_path (Path): Path to the input DAE file.
        output_path (Path): Path to save the output GLTF file.

    Returns:
        Path: Path to the converted GLTF file.
    """
    # Use Blender to convert with correct coordinate system.
    blender_script = f"""
import bpy

# Clear existing objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Import DAE (COLLADA)
bpy.ops.wm.collada_import(
    filepath='{input_path.as_posix()}'
)

# Select all objects
bpy.ops.object.select_all(action='SELECT')

# Export as glTF
bpy.ops.export_scene.gltf(
    filepath='{output_path.as_posix()}',
    export_format='GLTF_SEPARATE',
    use_selection=True,
    export_yup=True
)
"""

    # Write to a *non-deleted* temporary file.
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(blender_script)
        script_path = f.name

    try:
        # Run Blender.
        result = subprocess.run(
            [
                "env",
                "-u",
                "LD_LIBRARY_PATH",
                "blender",
                "--background",
                "--python",
                script_path,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Blender conversion failed:\n{result.stderr}\n{result.stdout}"
            )
    finally:
        # Clean up temporary script.
        os.remove(script_path)

    return output_path


def convert_usd_to_gltf(input_path: Path, output_path: Path) -> Path:
    """Convert USD/USDA/USDZ file to GLTF format using Blender.

    This ensures the model is in a right-handed coordinate system with:
    - +Y as up
    - +X as right
    - -Z as forward

    Args:
        input_path (Path): Path to the input USD file.
        output_path (Path): Path to save the output GLTF file.

    Returns:
        Path: Path to the converted GLTF file.
    """
    # Use Blender to convert with correct coordinate system.
    blender_script = f"""
import bpy

# Clear existing objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Import USD
bpy.ops.wm.usd_import(
    filepath='{input_path.as_posix()}'
)

# Select all objects
bpy.ops.object.select_all(action='SELECT')

# Export as glTF
bpy.ops.export_scene.gltf(
    filepath='{output_path.as_posix()}',
    export_format='GLTF_SEPARATE',
    use_selection=True,
    export_yup=True
)
"""

    # Write to a *non-deleted* temporary file.
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(blender_script)
        script_path = f.name

    try:
        # Run Blender.
        result = subprocess.run(
            [
                "env",
                "-u",
                "LD_LIBRARY_PATH",
                "blender",
                "--background",
                "--python",
                script_path,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Blender conversion failed:\n{result.stderr}\n{result.stdout}"
            )
    finally:
        # Clean up temporary script.
        os.remove(script_path)

    return output_path


def convert_glb_to_obj(input_path: Path, output_path: Path) -> Path:
    """Convert GLB or GLTF file to OBJ format using Blender.

    Args:
        input_path (Path): Path to the input GLB file.
        output_path (Path): Path to save the output OBJ file.

    Returns:
        Path: Path to the converted OBJ file.
    """
    blender_script = f"""
import bpy

# Clear existing objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Import GLB or GLTF
bpy.ops.import_scene.gltf(filepath='{input_path.as_posix()}')

# Select all objects
bpy.ops.object.select_all(action='SELECT')

# Export as OBJ
bpy.ops.wm.obj_export(
    filepath='{output_path.as_posix()}',
    export_selected_objects=True,
    export_uv=True,
    export_normals=True,
    export_materials=True,
    forward_axis='Y',
    up_axis='Z',
    path_mode='COPY',
    export_pbr_extensions=True
)
"""

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(blender_script)
        script_path = f.name

    try:
        result = subprocess.run(
            [
                "env",
                "-u",
                "LD_LIBRARY_PATH",
                "blender",
                "--background",
                "--python",
                script_path,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Blender conversion failed:\n{result.stderr}\n{result.stdout}"
            )
    finally:
        os.remove(script_path)

    return output_path


def convert_glb_to_gltf(input_path: Path, output_path: Path) -> Path:
    """Convert GLB or GLTF file to GLTF with separate files format using Blender.

    Args:
        input_path (Path): Path to the input GLB file.
        output_path (Path): Path to save the output GLTF file.

    Returns:
        Path: Path to the converted GLTF file.
    """
    blender_script = f"""
import bpy

# Clear existing objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Import GLB or GLTF
bpy.ops.import_scene.gltf(filepath='{input_path.as_posix()}')

# Select all objects
bpy.ops.object.select_all(action='SELECT')

# Export as glTF
bpy.ops.export_scene.gltf(
    filepath='{output_path.as_posix()}',
    export_format='GLTF_SEPARATE',
    use_selection=True,
    export_yup=True
)
"""

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(blender_script)
        script_path = f.name

    try:
        result = subprocess.run(
            [
                "env",
                "-u",
                "LD_LIBRARY_PATH",
                "blender",
                "--background",
                "--python",
                script_path,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Blender conversion failed:\n{result.stderr}\n{result.stdout}"
            )
    finally:
        os.remove(script_path)

    return output_path


def combine_gltf_files(
    gltf_paths: list[Path], output_path: Path, poses: list[np.ndarray] | None = None
) -> None:
    """Combine multiple GLTF files into a single GLTF file.

    Args:
        gltf_paths: List of paths to the GLTF files.
        output_path: Path to save the output GLTF file.
        poses: List of poses for each GLTF file. Each pose should be a 4x4
            transformation matrix. If None, no poses will be applied.
    """

    if not gltf_paths:
        raise ValueError("At least one GLTF file must be provided")
    if poses is None:
        poses = [np.eye(4) for _ in gltf_paths]
    elif len(gltf_paths) != len(poses):
        raise ValueError("Number of GLTF files must match number of poses")

    # Build the Blender script.
    blender_script = """
import bpy
import mathutils

# Clear existing objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Import and transform each GLTF file
"""

    for i, (gltf_path, pose_matrix) in enumerate(zip(gltf_paths, poses)):
        # Extract translation from the transformation matrix
        x, y, z = pose_matrix[:3, 3]

        # Extract rotation matrix and convert to Euler angles
        rotation_matrix = pose_matrix[:3, :3]
        rotation = Rotation.from_matrix(rotation_matrix)
        rx, ry, rz = rotation.as_euler("xyz")

        blender_script += f"""
# Import GLTF file {i+1}: {gltf_path.name}
bpy.ops.import_scene.gltf(filepath='{gltf_path.as_posix()}')

# Get all objects that were just imported (they should be selected)
imported_objects = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']

# Process each imported object to apply the pose transformation
for obj in imported_objects:
    # Clear selection and select only this object
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    
    # Apply current transforms to clear any existing transforms
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    
    # Set both rotation and translation directly
    obj.rotation_euler = mathutils.Euler(({rx}, {ry}, {rz}), 'XYZ')
    obj.location = mathutils.Vector(({x}, {y}, {z}))
    
    # Apply both transformations to bake them into the geometry
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=False)

# Deselect all objects to prepare for the next import
bpy.ops.object.select_all(action='DESELECT')

"""

    blender_script += f"""
# Select all objects for export
bpy.ops.object.select_all(action='SELECT')

# Export as glTF
bpy.ops.export_scene.gltf(
    filepath='{output_path.as_posix()}',
    export_format='GLTF_SEPARATE',
    use_selection=True,
    export_yup=True
)
"""

    # Write to a temporary file.
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(blender_script)
        script_path = f.name

    try:
        # Run Blender.
        result = subprocess.run(
            [
                "env",
                "-u",
                "LD_LIBRARY_PATH",
                "blender",
                "--background",
                "--python",
                script_path,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Blender combination failed:\n{result.stderr}\n{result.stdout}"
            )
    finally:
        # Clean up temporary script.
        os.remove(script_path)

    # Remove extensionsRequired from the output GLTF for Drake/VTK compatibility.
    remove_gltf_extensions_required(output_path)
