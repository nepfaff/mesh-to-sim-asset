import os
import subprocess
import tempfile

from pathlib import Path


def run_rolopoly(
    input_path: Path, output_folder: Path, timeout: int = 600, screen_size: int = 100
):
    """Run RoLoPoly surface remeshing on the input mesh.

    Args:
        input_path (Path): Path to the input mesh file.
        output_folder (Path): Path to the output folder for the remeshed mesh.
        timeout (int): Timeout in seconds to prevent indefinite blocking.
        screen_size (int): The RoLoPoly screen size. A smaller value will result in a
            simpler mesh. 100 is a good starting point. For more complex meshes, 200/300
            may be needed.

    Raises:
        RuntimeError: If RoLoPoly fails to process the mesh or Wine encounters an error.
    """
    try:
        # Set up environment to run Wine headless and suppress GUI dialogs.
        env = {
            **subprocess.os.environ,
            "DISPLAY": "",  # No display to prevent GUI dialogs
            "WINEDEBUG": "-all",  # Suppress Wine debug messages
            "WINE_ALLOW_XIM": "0",  # Disable X Input Method
        }
        subprocess.run(
            [
                "wine",
                "RoLoPoly/SurfaceRemeshingCli_bin.exe",
                "-i",
                input_path.as_posix(),
                "-n",
                str(screen_size),
                "-o",
                output_folder.as_posix(),
            ],
            check=True,
            capture_output=True,
            encoding="utf-8",
            timeout=timeout,
            env=env,  # Use modified environment
        )

    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"RoLoPoly failed with exit code {e.returncode}: "
            f"{e.stderr.strip() if e.stderr else 'No error message'}"
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"RoLoPoly timed out after {timeout} seconds. This may indicate a Wine "
            "popup or hanging process."
        )
    except FileNotFoundError:
        raise RuntimeError(
            "Wine or RoLoPoly executable not found. Make sure Wine is installed and "
            "RoLoPoly is in the correct path."
        )


def simplify_mesh_non_destructive(
    input_path: Path, output_path: Path, merge_distance: float = 0.001
):
    """Clean up a mesh using Blender by merging by distance and deleting loose geometry.

    This function performs non-destructive mesh cleanup that preserves the overall
    mesh structure while removing small gaps and loose vertices/edges/faces.

    Args:
        input_path (Path): Path to the input mesh file.
        output_path (Path): Path where the cleaned mesh will be saved.
        merge_distance (float): Distance threshold for merging vertices in meters.
            Default is 0.001 (1mm).

    Raises:
        RuntimeError: If Blender fails to process the mesh or is not found.
    """
    # Create a temporary Blender script to perform the cleanup operations.
    blender_script = f"""
import bpy
import bmesh

# Clear existing mesh data
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# Import the mesh
bpy.ops.import_mesh.stl(filepath="{input_path.as_posix()}")

# Get the active object (should be the imported mesh)
obj = bpy.context.active_object
if obj is None or obj.type != 'MESH':
    raise RuntimeError("No mesh object found after import")

# Enter Edit mode
bpy.context.view_layer.objects.active = obj
bpy.ops.object.mode_set(mode='EDIT')

# Select all geometry
bpy.ops.mesh.select_all(action='SELECT')

# Merge vertices by distance
bpy.ops.mesh.remove_doubles(threshold={merge_distance})

# Delete loose geometry (vertices, edges, faces not connected to faces)
bpy.ops.mesh.delete_loose()

# Limited dissolve to reduce polygon count while preserving shape
# 1 degree threshold (converted to radians)
import math
bpy.ops.mesh.dissolve_limited(angle_limit=math.radians(1.0))

# Return to Object mode
bpy.ops.object.mode_set(mode='OBJECT')

# Export the cleaned mesh
bpy.ops.export_mesh.stl(filepath="{output_path.as_posix()}")

print("Mesh cleanup completed successfully")
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
                f"Blender mesh cleanup failed:\n{result.stderr}\n{result.stdout}"
            )
    finally:
        # Clean up temporary script.
        os.remove(script_path)
