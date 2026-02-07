"""
Tools for computing physics properties of objects and creating SDF files with physics
simulation results.
"""

import argparse
import base64
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time

from contextlib import nullcontext
from importlib import resources
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import openai
import pyvista as pv
import trimesh
import yaml

from PIL import Image, ImageDraw, ImageFont

from mesh_to_sim_asset.mesh_conversion import combine_gltf_files

# Set up logger.
logger = logging.getLogger(__name__)


def _read_system_prompt(system_prompt_path: str | Path) -> str:
    """Read system prompt from file or package data.

    Args:
        system_prompt_path: Path to the system prompt file.

    Returns:
        Content of the system prompt file.
    """
    if (
        isinstance(system_prompt_path, str)
        and not os.path.isabs(system_prompt_path)
        and not os.path.exists(system_prompt_path)
    ):
        # Try to load from package data if it's a relative path that doesn't exist.
        try:
            from mesh_to_sim_asset import data

            return (resources.files(data) / system_prompt_path).read_text(
                encoding="utf-8"
            )
        except (ImportError, FileNotFoundError):
            # Fall back to original behavior.
            pass

    with open(system_prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def _resolve_blender_env_path(env_path: str | Path | None) -> str | None:
    """Resolve path to Blender environment file from package data or filesystem.

    Args:
        env_path: Path to the Blender environment file (.blend).

    Returns:
        Resolved path to the environment file, or None if env_path is None.
    """
    if env_path is None:
        return None

    env_path_str = str(env_path)

    if (
        isinstance(env_path, str)
        and not os.path.isabs(env_path)
        and not os.path.exists(env_path)
    ):
        # Try to load from package data if it's a relative path that doesn't exist.
        try:
            from mesh_to_sim_asset import data

            package_file_path = resources.files(data) / env_path
            if package_file_path.is_file():
                return str(package_file_path)
        except (ImportError, FileNotFoundError):
            # Fall back to original behavior.
            pass

    return env_path_str


def _resolve_materials_path(materials_path: str | Path) -> str:
    """Resolve path to materials YAML file from package data or filesystem.

    Args:
        materials_path: Path to the materials YAML file.

    Returns:
        Resolved path to the materials file.
    """
    materials_path_str = str(materials_path)

    if (
        isinstance(materials_path, str)
        and not os.path.isabs(materials_path)
        and not os.path.exists(materials_path)
    ):
        # Try to load from package data if it's a relative path that doesn't exist.
        try:
            from mesh_to_sim_asset import data

            package_file_path = resources.files(data) / materials_path
            if package_file_path.is_file():
                return str(package_file_path)
        except (ImportError, FileNotFoundError):
            # Fall back to original behavior.
            pass

    return materials_path_str


def encode_image_to_base64(image: np.ndarray | str) -> str:
    """Encodes an image to a base64 string.

    Args:
        image: Either a numpy array of shape (H, W, 3) in RGB format or a path to an
            image file.

    Returns:
        str: The base64 encoded image string.
    """
    if isinstance(image, str):
        # Read image directly from path.
        with Image.open(image) as img:
            # Convert to RGB in case it's not.
            img = img.convert("RGB")
            # Save to bytes.
            buffer = BytesIO()
            img.save(buffer, format="JPEG")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
    else:
        # Convert numpy array to PIL Image.
        img = Image.fromarray(image)
        buffer = BytesIO()
        img.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


def add_number_overlay(image_path: str, number: int, prefix: str = "") -> None:
    """Add a white background with black number overlay to an image.

    Args:
        image_path: Path to the image file to modify.
        number: Number to display on the overlay.
        prefix: Prefix to add to the image label.
    """
    with Image.open(image_path) as img:
        # Create a drawing context.
        draw = ImageDraw.Draw(img)

        # Try to use a smaller font to avoid hiding mesh, fallback to default.
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20
            )
        except (OSError, IOError):
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except (OSError, IOError):
                font = ImageFont.load_default()

        # Get text and measure its dimensions.
        text = f"{prefix}{number}"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Define overlay position and calculate background size based on text.
        margin = 10
        padding = 8  # Padding around text inside the background rectangle
        bg_width = text_width + 2 * padding
        bg_height = text_height + 2 * padding

        # Draw white background rectangle.
        draw.rectangle(
            [margin, margin, margin + bg_width, margin + bg_height],
            fill="white",
            outline="black",
            width=2,
        )

        # Position text centered in the rectangle.
        text_x = margin + padding
        text_y = margin + padding

        # Draw the label in black.
        draw.text((text_x, text_y), text, fill="black", font=font)

        # Save the modified image.
        img.save(image_path)


def get_view_direction_from_image_number(
    image_number: int, num_side_views: int = 8, include_diagonal_views: bool = False
) -> str:
    """Map an image number to its corresponding view direction and axis.

    Args:
        image_number: The image number (0-based index).
        num_side_views: Number of side views rendered.
        include_diagonal_views: Whether diagonal views are included.

    Returns:
        String representing the view axis: "x", "-y", "z", "top", "bottom",
        "top_diagonal", "bottom_diagonal"
    """
    if image_number == 0:
        return "top"
    elif image_number == 1:
        return "bottom"
    elif 2 <= image_number < 2 + num_side_views:
        # Side views are rendered in a circle around the XY plane.
        side_index = image_number - 2
        angle = 2 * np.pi * side_index / num_side_views

        # Map to closest primary axis.
        if angle <= np.pi / 4 or angle > 7 * np.pi / 4:
            return "x"
        elif np.pi / 4 < angle <= 3 * np.pi / 4:
            return "y"
        elif 3 * np.pi / 4 < angle <= 5 * np.pi / 4:
            return "-x"
        else:  # 5*np.pi/4 < angle <= 7*np.pi/4
            return "-y"
    elif include_diagonal_views and image_number == 2 + num_side_views:
        return "top_diagonal"
    elif include_diagonal_views and image_number == 3 + num_side_views:
        return "bottom_diagonal"
    else:
        max_expected = 1 + num_side_views + (2 if include_diagonal_views else 0)
        raise ValueError(
            f"Invalid image number {image_number}. Expected 0 to {max_expected}"
        )


def get_front_axis_from_image_number(
    image_number: int, num_side_views: int = 8, include_diagonal_views: bool = False
) -> str:
    """Get the coordinate axis that corresponds to the front direction based on the
    image number.

    Args:
        image_number: The image number that the LLM identified as the front view.
        num_side_views: Number of side views rendered.
        include_diagonal_views: Whether diagonal views are included.

    Returns:
        String representing the front axis: "x", "-x", "y", "-y", "z", or "-z"
    """
    view_axis = get_view_direction_from_image_number(
        image_number=image_number,
        num_side_views=num_side_views,
        include_diagonal_views=include_diagonal_views,
    )

    # Map view direction to front axis.
    if view_axis == "top":
        return "z"
    elif view_axis == "bottom":
        return "-z"
    elif view_axis == "top_diagonal":
        return "z"  # Diagonal from top, still primarily z-direction
    elif view_axis == "bottom_diagonal":
        return "-z"  # Diagonal from bottom, still primarily -z-direction
    return view_axis


def render_mesh_views(
    gltf_path: str | Path,
    num_side_views: int = 8,
    env_path: str | None = None,
    output_dir: Path | None = None,
    use_cpu_rendering: bool = False,
    image_label_prefix: str = "",
    include_diagonal_views: bool = False,
) -> list[str]:
    """Render a GLTF mesh from multiple views using Blender.

    Args:
        gltf_path (str): Path to the GLTF mesh file.
        num_side_views (int): Number of equidistant side views to render.
        env_path (str | None): Path to a Blender environment file (.blend) containing
            lighting and material setup. If None, a basic environment will be created.
        output_dir (Path | None): Directory to save rendered images. If None, a
            temporary directory will be created and returned. Note: If a temporary
            directory is created, the caller is responsible for cleaning it up.
        use_cpu_rendering (bool): If True, forces CPU rendering. If False, uses GPU
            rendering if available.
        image_label_prefix (str): Prefix to add to the image labels.
        include_diagonal_views (bool): If True, includes two additional views at 45
            degrees from the top and bottom views.

    Returns:
        list[str]: List of paths to the rendered images.
    """
    # Create a temporary directory for the rendered images if no output directory is
    # provided.
    temp_dir_context = (
        tempfile.TemporaryDirectory() if output_dir is None else nullcontext()
    )
    temp_dir = Path(temp_dir_context.name) if output_dir is None else output_dir

    # Resolve env_path to handle package resources.
    resolved_env_path = _resolve_blender_env_path(env_path)

    # GPU OptiX setup block for CYCLES rendering.
    gpu_optix_setup = """
# Enable GPU OptiX rendering.
prefs = bpy.context.preferences.addons['cycles'].preferences
prefs.compute_device_type = 'OPTIX'
prefs.get_devices()
for device in prefs.devices:
    device.use = device.type == 'OPTIX' or device.type == 'CPU'
bpy.context.scene.cycles.device = 'GPU'
# Enable persistent data to keep BVH in memory between renders.
bpy.context.scene.render.use_persistent_data = True
"""

    # Create Blender script to render views.
    if resolved_env_path:
        env_block = f"""
bpy.ops.wm.open_mainfile(filepath='{resolved_env_path}')
# Clear any existing objects from the environment.
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.cycles.samples = 32
scene.render.resolution_x = 512
scene.render.resolution_y = 512
{gpu_optix_setup}
"""
    else:
        env_block = f"""
# Set up basic environment.
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.cycles.samples = 32
scene.render.resolution_x = 256
scene.render.resolution_y = 256
{gpu_optix_setup}

# Add a light.
light = bpy.data.lights.new(name="Light", type='SUN')
light_obj = bpy.data.objects.new("Light", light)
scene.collection.objects.link(light_obj)
light_obj.rotation_euler = (math.radians(45), 0, math.radians(45))
"""

    # Add CPU rendering configuration if requested.
    cpu_config = ""
    if use_cpu_rendering:
        cpu_config = """
# Force CPU rendering
bpy.context.scene.cycles.device = 'CPU'
"""

    blender_script = f"""
import bpy
import math
import os
import json
from mathutils import Vector

# Clear scene.
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

{env_block}

{cpu_config}

# Import GLTF.
bpy.ops.import_scene.gltf(filepath='{gltf_path}')
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

# Compute bounding box.
mesh_objs = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']
bbox_min = Vector((float('inf'),) * 3)
bbox_max = Vector((float('-inf'),) * 3)
for obj in mesh_objs:
    for corner in obj.bound_box:
        world_corner = obj.matrix_world @ Vector(corner)
        bbox_min = Vector(map(min, bbox_min, world_corner))
        bbox_max = Vector(map(max, bbox_max, world_corner))
bbox_center = (bbox_min + bbox_max) / 2
bbox_size = bbox_max - bbox_min
max_dim = max(bbox_size)

# Setup camera.
scene = bpy.context.scene
camera = bpy.data.cameras.new(name="Camera")
camera_obj = bpy.data.objects.new("Camera", camera)
scene.collection.objects.link(camera_obj)
scene.camera = camera_obj
camera.type = 'PERSP'
camera.lens = 50
camera.sensor_width = 36
camera.clip_start = 0.01
camera.clip_end = 100000

# Compute camera base distance.
fov = 2 * math.atan((camera.sensor_width / 2) / camera.lens)
base_distance = (max_dim / 2) / math.tan(fov / 2)
margin_scale = 1.5  # Add margin to camera distance

# Look-at function.
def look_at(obj, target):
    direction = (target - obj.location).normalized()
    quat = direction.to_track_quat('-Z', 'Y')
    obj.rotation_euler = quat.to_euler()

# Axis material helper.
def make_material(name, rgba):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = rgba
    return mat

# Create a labeled arrow axis.
def create_arrow_axis(name, direction: Vector, color, label_text, scale=1.0):
    # Scale arrow dimensions based on object size
    base_scale = max_dim * 0.01  # Base scale for thickness
    shaft_len = max_dim * 0.6 * scale
    tip_len = max_dim * 0.15 * scale
    tip_radius = base_scale * 2.5  # Thinner tip
    shaft_radius = base_scale * 1.0  # Thinner shaft
    mat = make_material(f"{{name}}_mat", color)

    # Shaft
    bpy.ops.mesh.primitive_cylinder_add(radius=shaft_radius, depth=shaft_len, location=bbox_center)
    shaft = bpy.context.active_object
    shaft.name = f"{{name}}_shaft"
    shaft.data.materials.append(mat)
    shaft.location = bbox_center + (direction.normalized() * shaft_len / 2)
    shaft.rotation_mode = 'QUATERNION'
    shaft.rotation_quaternion = direction.to_track_quat('Z', 'Y')

    # Tip
    bpy.ops.mesh.primitive_cone_add(radius1=tip_radius, depth=tip_len)
    tip = bpy.context.active_object
    tip.name = f"{{name}}_tip"
    tip.data.materials.append(mat)
    tip.location = bbox_center + (direction.normalized() * (shaft_len + tip_len / 2))
    tip.rotation_mode = 'QUATERNION'
    tip.rotation_quaternion = direction.to_track_quat('Z', 'Y')

    # Label
    bpy.ops.object.text_add(location=tip.location + direction.normalized() * (tip_len * 0.8))
    text = bpy.context.active_object
    text.data.body = label_text
    text.scale = (base_scale * 5,) * 3  # Scale text size with object
    text.data.extrude = base_scale * 0.5  # Scale text depth with object
    text.data.align_x = 'CENTER'
    text.data.align_y = 'CENTER'
    text.rotation_mode = 'QUATERNION'
    text.rotation_quaternion = direction.to_track_quat('Z', 'Y')
    text_mat = make_material(f"{{name}}_label", (1, 1, 1, 1))  # white text
    text.data.materials.append(text_mat)

# Create coordinate frame.
create_arrow_axis("X", Vector((1, 0, 0)), (1, 0, 0, 1), "+X")
create_arrow_axis("Y", Vector((0, 1, 0)), (0, 1, 0, 1), "+Y")
create_arrow_axis("Z", Vector((0, 0, 1)), (0, 0, 1, 1), "+Z")

# Views.
views = []
views.append({{"name": "0_top", "direction": Vector((0, 0, 1))}})
views.append({{"name": "1_bottom", "direction": Vector((0, 0, -1))}})
for i in range({num_side_views}):
    angle = 2 * math.pi * i / {num_side_views}
    dir_vec = Vector((math.cos(angle), math.sin(angle), 0))
    views.append({{"name": f"{{i}}_side", "direction": dir_vec}})

# Add diagonal views if enabled.
if {include_diagonal_views}:
    # 45-degree view from top (tilted towards +X direction).
    top_diagonal = Vector((1, 0, 1)).normalized()
    views.append({{"name": f"{2 + num_side_views}_top_diagonal", "direction": top_diagonal}})
    
    # 45-degree view from bottom (tilted towards +X direction).
    bottom_diagonal = Vector((1, 0, -1)).normalized()
    views.append({{"name": f"{3 + num_side_views}_bottom_diagonal", "direction": bottom_diagonal}})

# Render each view.
image_paths = []
for view in views:
    direction = view["direction"].normalized()
    camera_distance = base_distance * margin_scale
    camera_obj.location = bbox_center + direction * camera_distance
    look_at(camera_obj, bbox_center)
    scene.render.filepath = os.path.join("{temp_dir}", f"{{view['name']}}.png")
    bpy.ops.render.render(write_still=True)
    image_paths.append(scene.render.filepath)

# Write paths to JSON file.
with open(os.path.join("{temp_dir}", "image_paths.json"), "w", encoding="utf-8") as f:
    json.dump(image_paths, f)
"""
    # Write script to temporary file.
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
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Blender rendering failed:\n{result.stderr}\n{result.stdout}"
            )

        # Read paths from JSON file.
        json_path = temp_dir / "image_paths.json"
        if not json_path.exists():
            raise RuntimeError(
                f"Blender did not create the expected JSON file at {json_path}. "
                f"Blender output:\nstdout: {result.stdout}\nstderr: {result.stderr}"
            )

        with open(json_path, "r", encoding="utf-8") as f:
            image_paths = json.load(f)

        # Add number overlays to each image.
        for i, image_path in enumerate(image_paths):
            add_number_overlay(
                image_path=image_path, number=i, prefix=image_label_prefix
            )

            # Rename image to include the number.
            new_name = f"{image_label_prefix}_{i}.png"
            new_path = os.path.join(output_dir, new_name)
            os.rename(image_path, new_path)
            image_paths[i] = new_path

        return image_paths

    finally:
        # Clean up temporary files.
        os.remove(script_path)
        # Clean up the JSON file.
        os.remove(temp_dir / "image_paths.json")


def analyze_mesh_physics(
    gltf_path: str | Path,
    system_prompt_path: str | Path = "physics_system_prompt.txt",
    num_side_views: int = 8,
    model: str = "o3-2025-04-16",
    env_path: str | None = "studio.blend",
    metadata: str | None = None,
    images_path: str | None = None,
    use_cpu_rendering: bool = False,
) -> dict[str, Any]:
    """Analyze a GLTF mesh's physical properties using OpenAI's API.
    Expects the OPENAI_API_KEY environment variable to be set.

    Args:
        gltf_path (str): Path to the GLTF mesh file.
        system_prompt_path (str): Path to the system prompt file.
        num_side_views (int): Number of equidistant side views to render.
        model (str): OpenAI model to use.
        env_path (str | None): Path to a Blender environment file (.blend) containing
            lighting and material setup. If None, a basic environment will be created.
        metadata (str | None): Optional metadata about the object (e.g., dimensions,
            category, description) to include in the analysis.
        images_path (str | None): Path to the directory to save the image renders to. If
            None, they will be saved in a temporary directory and deleted after the
            analysis.
        use_cpu_rendering (bool): If True, forces CPU rendering. If False, uses GPU
            rendering if available.
    Returns:
        dict[str, Any]: Dictionary containing the physical properties analysis.

    Raises:
        RuntimeError: If OPENAI_API_KEY environment variable is not set.
    """
    # Get API key from environment variable.
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY environment variable is not set. "
            "Please set it using:\n"
            "    export OPENAI_API_KEY='your-api-key-here'\n"
            "You can get an API key from https://platform.openai.com/api-keys"
        )

    # Read system prompt.
    system_prompt = _read_system_prompt(system_prompt_path)

    try:
        # Set up output directory.
        if images_path is not None:
            output_dir = Path(images_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            temp_dir_to_cleanup = None
        else:
            temp_dir_to_cleanup = tempfile.mkdtemp()
            output_dir = Path(temp_dir_to_cleanup)

        # Render mesh views.
        image_paths = render_mesh_views(
            gltf_path=gltf_path,
            num_side_views=num_side_views,
            env_path=env_path,
            output_dir=output_dir,
            use_cpu_rendering=use_cpu_rendering,
            include_diagonal_views=True,
        )

        # Prepare images for API.
        images = []
        for path in image_paths:
            images.append(encode_image_to_base64(path))

        # Prepare user message with optional metadata.
        user_message = (
            "Please analyze these multi-view renders of the object "
            f"(asset name {gltf_path.stem})"
        )
        if metadata:
            user_message += f" with the following metadata:\n{metadata}\n"
        user_message += (
            " and provide the physical properties analysis in the specified "
            "JSON format."
        )

        # Save the user message to a file.
        vlm_user_message_path = output_dir.parent / "vlm_user_message.txt"
        with open(vlm_user_message_path, "w", encoding="utf-8") as f:
            f.write(user_message)

        # Call OpenAI API with retry on empty response.
        client = openai.OpenAI(api_key=api_key)
        thinking_args = {"reasoning_effort": "low"} if "o" in model else {}
        max_retries = 3
        for attempt in range(max_retries):
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": user_message,
                            },
                            *[
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{img}"
                                    },
                                }
                                for img in images
                            ],
                        ],
                    },
                ],
                response_format={"type": "json_object"},
                **thinking_args,
            )

            # Check for valid response.
            json_str = response.choices[0].message.content
            if json_str is not None:
                return json.loads(json_str)

            # Retry with exponential backoff.
            if attempt < max_retries - 1:
                wait_time = 2**attempt
                logger.warning(
                    f"OpenAI API returned empty response, retrying in {wait_time}s "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(wait_time)

        raise ValueError(
            "OpenAI API returned empty response content after all retries. "
            "This may be due to content filtering, rate limiting, or model refusal."
        )
    finally:
        # Clean up temporary directory if one was created.
        if temp_dir_to_cleanup is not None:
            shutil.rmtree(temp_dir_to_cleanup)


def analyze_composed_asset_physics(
    gltf_paths: list[Path],
    gltf_poses: list[np.ndarray] | None = None,
    combined_gltf_path: Path | None = None,
    system_prompt_path: str | Path = "composed_physics_system_prompt.txt",
    num_side_views: int = 8,
    model: str = "o3-2025-04-16",
    env_path: str | None = "studio.blend",
    metadata: str | None = None,
    images_path: str | None = None,
    use_cpu_rendering: bool = False,
) -> dict[str, Any]:
    """Analyze the physical properties of an asset consisting of multiple sub-meshes.

    Args:
        gltf_paths: List of paths to the GLTF mesh files that make up the asset.
        gltf_poses: List of poses for each GLTF file. Each pose should be a 4x4
            transformation matrix. If None, no poses will be applied.
        combined_gltf_path: Path to the combined GLTF file. If None, the GLTF files
            will be combined into a single file in the same directory as the first
            GLTF file.
        system_prompt_path: Path to the system prompt file.
        num_side_views: Number of equidistant side views to render for each GLTF file.
        model: OpenAI model to use.
        env_path: Path to a Blender environment file (.blend).
        metadata: Optional metadata about the object.
        images_path: Path to the directory to save the image renders to. If None, they
            will be saved in a temporary directory and deleted after the analysis.
        use_cpu_rendering: If True, forces CPU rendering. If False, uses GPU
            rendering if available.

    Returns:
        The properties for the entire asset, including the properties for the sub-parts.
    """
    # Get API key from environment variable.
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY environment variable is not set. "
            "Please set it using:\n"
            "    export OPENAI_API_KEY='your-api-key-here'\n"
            "You can get an API key from https://platform.openai.com/api-keys"
        )

    if combined_gltf_path is None:
        # Combine the GLTF files.
        combined_gltf_path = gltf_paths[0].parent / "combined_scene.gltf"
        combine_gltf_files(
            gltf_paths=gltf_paths, poses=gltf_poses, output_path=combined_gltf_path
        )

    # Read system prompt.
    system_prompt = _read_system_prompt(system_prompt_path)

    try:
        # Set up output directory.
        if images_path is not None:
            output_dir = Path(images_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            temp_dir_to_cleanup = None
        else:
            temp_dir_to_cleanup = tempfile.mkdtemp()
            output_dir = Path(temp_dir_to_cleanup)

        # Render mesh views.
        image_paths = render_mesh_views(
            gltf_path=combined_gltf_path,
            num_side_views=num_side_views,
            env_path=env_path,
            output_dir=output_dir,
            use_cpu_rendering=use_cpu_rendering,
            image_label_prefix="combined_#",
            include_diagonal_views=True,
        )
        for i, gltf_path in enumerate(gltf_paths):
            image_paths.extend(
                render_mesh_views(
                    gltf_path=gltf_path,
                    num_side_views=num_side_views // 2,  # Less views for subparts
                    env_path=env_path,
                    output_dir=output_dir,
                    use_cpu_rendering=use_cpu_rendering,
                    image_label_prefix=f"part_{i}_#",
                    include_diagonal_views=False,  # Don't include diagonal views for subparts
                )
            )

        # Prepare images for API.
        images = []
        for path in image_paths:
            images.append(encode_image_to_base64(path))

        # Add individual asset dimensions to metadata.
        if metadata is None:
            metadata = ""
        else:
            metadata += "\n"
        metadata += "Individual sub-parts dimensions:\n"
        for i, gltf_path in enumerate(gltf_paths):
            mesh = trimesh.load_mesh(gltf_path)
            # Swap y and z to match Blender's coordinate system from rendering.
            d_x, d_z, d_y = mesh.bounding_box.extents
            metadata += (
                f"Part {i} (part_{i}_#, part file name {gltf_path.stem}): {d_x}m x "
                f"{d_y}m x {d_z}m (x, y, z axes respectively).\n"
            )

        # Prepare user message with optional metadata.
        user_message = (
            "Please analyze these multi-view renders of the object "
            f"(asset name {output_dir.parent.stem})"
        )
        if metadata:
            user_message += f" with the following metadata:\n{metadata}"
        user_message += (
            " and provide the physical properties analysis in the specified "
            "JSON format."
        )

        # Save the user message to a file.
        vlm_user_message_path = output_dir.parent / "vlm_user_message.txt"
        with open(vlm_user_message_path, "w", encoding="utf-8") as f:
            f.write(user_message)

        # Call OpenAI API with retry on empty response.
        client = openai.OpenAI(api_key=api_key)
        thinking_args = {"reasoning_effort": "low"} if "o" in model else {}
        max_retries = 3
        for attempt in range(max_retries):
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": user_message,
                            },
                            *[
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{img}"
                                    },
                                }
                                for img in images
                            ],
                        ],
                    },
                ],
                response_format={"type": "json_object"},
                **thinking_args,
            )

            # Check for valid response.
            json_str = response.choices[0].message.content
            if json_str is not None:
                return json.loads(json_str)

            # Retry with exponential backoff.
            if attempt < max_retries - 1:
                wait_time = 2**attempt
                logger.warning(
                    f"OpenAI API returned empty response, retrying in {wait_time}s "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(wait_time)

        raise ValueError(
            "OpenAI API returned empty response content after all retries. "
            "This may be due to content filtering, rate limiting, or model refusal."
        )
    finally:
        # Clean up temporary directory if one was created.
        if temp_dir_to_cleanup is not None:
            shutil.rmtree(temp_dir_to_cleanup)


def get_material_properties(
    material_name: str, materials_path: str | Path
) -> dict[str, float]:
    """Get properties for a specific material.

    Args:
        material_name: Name of the material to look up.
        materials_path: Path to the materials YAML file.

    Returns:
        Dictionary of material properties.

    Raises:
        KeyError: If material is not found.
    """
    # Resolve materials path to handle package resources.
    resolved_materials_path = _resolve_materials_path(materials_path)

    with open(resolved_materials_path, "r", encoding="utf-8") as f:
        materials = yaml.safe_load(f)

    if material_name not in materials:
        raise KeyError(f"Material '{material_name}' not found in materials database")

    # Convert any string values to float if they match scientific notation.
    props = {}
    for key, value in materials[material_name].items():
        if isinstance(value, str) and re.match(
            r"^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$", value
        ):
            props[key] = float(value)
        else:
            props[key] = value

    return props


def compute_hydroelastic_modulus(material_modulus: float, vtk_file_path: Path) -> float:
    """Computes the Hydroelastic based on the material modulus and mesh size. The
    material modulus is scale independent and leads to consistent surface stiffness,
    regardless of the mesh size. This allows us to get pressure as a function of depth
    in meters instead of as a percentage of the mesh size.

    For non-convex meshes, Drake computes the pressure field as `pressure = h * l(v)/L`,
    where v is a mesh vertex, l(v) is the distance from v to the closest point on the
    surface, and L is the max of l(v) over all v's.
    We then set `h = material_modulus * L` to get `pressure = material_modulus * l(v)`.

    See https://github.com/RobotLocomotion/drake/issues/23153 for more details.

    Args:
        material_modulus (float): The material modulus.
        vtk_file_path (Path): Path to the vtk file.

    Returns:
        float: The hydroelastic modulus.
    """
    suffix = vtk_file_path.suffix
    if suffix != ".vtk":
        raise ValueError(f"Expected a .vtk file, got {suffix}")

    volume_mesh = pv.read(vtk_file_path)

    # Extract surface from volume mesh.
    surface_mesh = volume_mesh.extract_surface()

    # Find closest points on surface.
    _, closest_points = surface_mesh.find_closest_cell(
        volume_mesh.points, return_closest_point=True
    )
    distances = np.linalg.norm(volume_mesh.points - closest_points, axis=1)  # l(v)

    # Compute the maximum distance.
    max_distance = float(np.max(distances))  # L

    # Compute the hydroelastic modulus.
    return material_modulus * max_distance


def calculate_scale_factor(
    mesh_path: str | Path,
    target_bbox: tuple[float, float, float],
) -> float:
    """Calculate the scale factor to match the LLM's predicted size.

    Args:
        mesh_path: Path to the mesh file.
        target_bbox: Target bounding box dimensions (width, height, depth) predicted by
            LLM.

    Returns:
        Scale factor to apply to match the LLM's predicted size.
    """
    # Get current bbox dimensions.
    mesh = trimesh.load(mesh_path)
    current_dims = mesh.bounding_box.extents

    # Swap y and z to match Blender's coordinate system from rendering.
    current_dims = np.array([current_dims[0], current_dims[2], current_dims[1]])

    # Calculate scale factors for each dimension.
    target_dims = np.array(target_bbox)
    scale_factors = target_dims / current_dims

    # Use the mean scale factor to match the LLM's predicted size.
    return float(np.mean(scale_factors))


def get_physics_properties(
    gltf_path: str | Path,
    system_prompt_path: str = "physics_system_prompt.txt",
    materials_path: str | Path = "materials.yaml",
    num_side_views: int = 8,
    model: str = "gpt-4.1-2025-04-14",
    env_path: str | None = "studio.blend",
    metadata: str | None = None,
    is_metric_scale: bool = False,
    images_path: str | None = None,
    use_cpu_rendering: bool = False,
) -> dict[str, Any]:
    """Get complete physics properties for a mesh by combining LLM analysis with
    material properties.

    Args:
        gltf_path: Path to the GLTF mesh file.
        system_prompt_path: Path to the system prompt file.
        materials_path: Path to the materials YAML file.
        num_side_views: Number of equidistant side views to render.
        model: OpenAI model to use.
        env_path: Path to a Blender environment file (.blend).
        metadata: Optional metadata about the object.
        is_metric_scale: If True, assumes mesh is already in metric scale and skips
            scaling.
        images_path: Path to the directory to save the image renders to. If None, they
            will be saved in a temporary directory and deleted after the analysis.
        use_cpu_rendering: If True, forces CPU rendering. If False, uses GPU
            rendering if available.

    Returns:
        Dictionary containing complete physics properties:
        - mass: Mass in kg (from LLM prediction)
        - mass_range: Mass range in kg from LLM prediction
        - mass_source: Source of mass determination from LLM prediction
        - material_modulus: Material modulus that determines pressure as a function of
            depth in meters
        - mu_dynamic: Dynamic friction coefficient
        - mu_static: Static friction coefficient
        - scale: Scale factor to fit target dimensions (1.0 if is_metric_scale is True)
        - name: Name of the object from LLM analysis
        - description: Description of the object from LLM analysis
        - material: Material of the object from LLM analysis
        - dimensions_m: Dimensions of the object in meters from LLM analysis
        - canonical_orientation: Canonical orientation of the object from LLM analysis
        - is_manipuland: Whether the object is manipulable
        - placement_options: Placement options for the object
        - is_single_object: Whether the object is a single or a collection of objects
        - big_with_thin_parts: Whether the object is big with thin parts (e.g., a table
            with thin legs, a chair)
        - asset_quality: Quality of the asset from LLM analysis
        - style: Style of the object from LLM analysis
        - location: Likely environment where the object is found from LLM analysis
        - is_textured: boolean, whether the mesh is textured/ has materials
        - is_simulatable: boolean, whether the mesh is simulatable
        - is_simulatable_reason: string, reason for `is_simulatable`
    """
    # Provide the mesh dimensions as metadata.
    mesh = trimesh.load_mesh(gltf_path)
    # Swap y and z to match Blender's coordinate system from rendering.
    d_x, d_z, d_y = mesh.bounding_box.extents
    if metadata is None:
        metadata = ""
    else:
        metadata += "\n"
    if is_metric_scale:
        metadata += (
            f"The mesh is already in metric units. The dimensions are {d_x}m, "
            f"{d_y}m, {d_z}m along the x, y, and z axes, respectively."
        )
    else:
        metadata += (
            f"The mesh may or may not be in metric units. The asset dimensions are "
            f"{d_x}m, {d_y}m, {d_z}m along the x, y, and z axes, respectively. "
            "Please make a sensible judgement on whether the mesh is in metric units "
            "or not. Feel free to use or discard these dimensions as you see fit."
        )

    # Get LLM analysis.
    llm_analysis = analyze_mesh_physics(
        gltf_path=gltf_path,
        system_prompt_path=system_prompt_path,
        num_side_views=num_side_views,
        model=model,
        env_path=env_path,
        metadata=metadata,
        images_path=images_path,
        use_cpu_rendering=use_cpu_rendering,
    )

    # Get material properties.
    props = get_material_properties(
        material_name=llm_analysis["material"], materials_path=materials_path
    )

    # Get friction.
    mu_dynamic = props["friction"]
    mu_static = mu_dynamic * 1.2  # Static friction is typically higher than dynamic

    # Calculate scale factor using target dimensions from LLM analysis.
    if is_metric_scale:
        scale = 1.0
    else:
        target_dims = llm_analysis["dimensions_m"]
        scale = calculate_scale_factor(gltf_path, target_dims)

    # Get front axis from LLM analysis.
    front_axis = get_front_axis_from_image_number(
        image_number=llm_analysis["canonical_orientation"]["front_view_image_index"],
        num_side_views=num_side_views,
        include_diagonal_views=True,
    )
    llm_analysis["canonical_orientation"]["front_axis"] = front_axis

    return {
        "mass": llm_analysis["mass_kg"],
        "mass_range": llm_analysis["mass_range_kg"],
        "mass_source": llm_analysis["mass_source"],
        "material_modulus": props["material_modulus"],
        "mu_dynamic": mu_dynamic,
        "mu_static": mu_static,
        "scale": scale,
        "name": llm_analysis["name"],
        "description": llm_analysis["description"],
        "material": llm_analysis["material"],
        "dimensions_m": llm_analysis["dimensions_m"],
        "canonical_orientation": llm_analysis["canonical_orientation"],
        "is_manipuland": llm_analysis["is_manipuland"],
        "placement_options": llm_analysis["placement_options"],
        "is_single_object": llm_analysis["is_single_object"],
        "big_with_thin_parts": llm_analysis["big_with_thin_parts"],
        "asset_quality": llm_analysis["asset_quality"],
        "style": llm_analysis["style"],
        "location": llm_analysis["location"],
        "is_textured": llm_analysis["is_textured"],
        "is_simulatable": llm_analysis["is_simulatable"],
        "is_simulatable_reason": llm_analysis["is_simulatable_reason"],
    }


def get_composed_asset_physics_properties(
    gltf_paths: list[Path],
    system_prompt_path: str = "composed_physics_system_prompt.txt",
    materials_path: str | Path = "materials.yaml",
    num_side_views: int = 8,
    model: str = "gpt-4.1-2025-04-14",
    env_path: str | None = "studio.blend",
    metadata: str | None = None,
    images_path: str | None = None,
    use_cpu_rendering: bool = False,
) -> dict[str, Any]:
    """Get complete physics properties for an asset consisting of multiple sub-meshes by
    combining LLM analysis with material properties. The sub-meshes are analyzed in
    context of the whole asset.

    Args:
        gltf_paths: List of paths to the GLTF mesh files that make up the asset.
        system_prompt_path: Path to the system prompt file.
        materials_path: Path to the materials YAML file.
        num_side_views: Number of equidistant side views to render.
        model: OpenAI model to use.
        env_path: Path to a Blender environment file (.blend).
        metadata: Optional metadata about the object.
        images_path: Path to the directory to save the image renders to. If None, they
            will be saved in a temporary directory and deleted after the analysis.
        use_cpu_rendering: If True, forces CPU rendering. If False, uses GPU
            rendering if available.

    Returns:
        Dictionary containing complete physics properties:
        - mass: Mass in kg (from LLM prediction)
        - mass_range: Mass range in kg from LLM prediction
        - mass_source: Source of mass determination from LLM prediction
        - material_modulus: Material modulus that determines pressure as a function of
            depth in meters
        - mu_dynamic: Dynamic friction coefficient
        - mu_static: Static friction coefficient
        - name: Name of the object from LLM analysis
        - description: Description of the object from LLM analysis
        - material: Material of the object from LLM analysis
        - dimensions_m: Dimensions of the object in meters from LLM analysis
        - canonical_orientation: Canonical orientation of the object from LLM analysis
        - is_manipuland: Whether the object is manipulable
        - placement_options: Placement options for the object
        - is_single_object: Whether the object is a single or a collection of objects
        - big_with_thin_parts: Whether the object is big with thin parts (e.g., a table
            with thin legs, a chair)
        - asset_quality: Quality of the asset from LLM analysis
        - style: Style of the object from LLM analysis
        - location: Likely environment where the object is found from LLM analysis
        - subparts: List of properties for each sub-part.
        - is_textured: boolean, whether the mesh is textured/ has materials
    """
    # Combine the GLTF files.
    combined_gltf_path = gltf_paths[0].parent / "combined_scene.gltf"
    combine_gltf_files(gltf_paths=gltf_paths, output_path=combined_gltf_path)

    # Provide the mesh dimensions as metadata.
    mesh = trimesh.load_mesh(combined_gltf_path)
    # Swap y and z to match Blender's coordinate system from rendering.
    d_x, d_z, d_y = mesh.bounding_box.extents
    if metadata is None:
        metadata = ""
    else:
        metadata += "\n"
    metadata += (
        f"The mesh is already in metric units. The dimensions are {d_x}m, "
        f"{d_y}m, {d_z}m along the x, y, and z axes, respectively."
    )

    # Get LLM analysis.
    llm_analysis = analyze_composed_asset_physics(
        gltf_paths=gltf_paths,
        combined_gltf_path=combined_gltf_path,
        system_prompt_path=system_prompt_path,
        num_side_views=num_side_views,
        model=model,
        env_path=env_path,
        metadata=metadata,
        images_path=images_path,
        use_cpu_rendering=use_cpu_rendering,
    )

    # Get material properties.
    props = get_material_properties(
        material_name=llm_analysis["material"], materials_path=materials_path
    )
    sub_part: dict[str, Any]
    for sub_part in llm_analysis["subparts"]:
        sub_part.update(
            get_material_properties(
                material_name=sub_part["material"], materials_path=materials_path
            )
        )

    # Get friction.
    mu_dynamic = props["friction"]
    mu_static = mu_dynamic * 1.2  # Static friction is typically higher than dynamic

    # Get front axis from LLM analysis.
    front_axis = get_front_axis_from_image_number(
        image_number=llm_analysis["canonical_orientation"]["front_view_image_index"],
        num_side_views=num_side_views,
        include_diagonal_views=True,
    )
    llm_analysis["canonical_orientation"]["front_axis"] = front_axis

    # Process subpart properties.
    for sub_part in llm_analysis["subparts"]:
        sub_part["mass"] = sub_part["mass_kg"]
        del sub_part["mass_kg"]

        # Friction.
        sub_part["mu_dynamic"] = sub_part["friction"]
        sub_part["mu_static"] = sub_part["friction"] * 1.2

    return {
        "mass": llm_analysis["mass_kg"],
        "mass_range": llm_analysis["mass_range_kg"],
        "mass_source": llm_analysis["mass_source"],
        "material_modulus": props["material_modulus"],
        "mu_dynamic": mu_dynamic,
        "mu_static": mu_static,
        "name": llm_analysis["name"],
        "description": llm_analysis["description"],
        "material": llm_analysis["material"],
        "dimensions_m": llm_analysis["dimensions_m"],
        "canonical_orientation": llm_analysis["canonical_orientation"],
        "is_manipuland": llm_analysis["is_manipuland"],
        "placement_options": llm_analysis["placement_options"],
        "is_single_object": llm_analysis["is_single_object"],
        "big_with_thin_parts": llm_analysis["big_with_thin_parts"],
        "asset_quality": llm_analysis["asset_quality"],
        "style": llm_analysis["style"],
        "location": llm_analysis["location"],
        "subparts": llm_analysis["subparts"],
        "is_textured": llm_analysis["is_textured"],
    }


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
        ],
    )

    parser = argparse.ArgumentParser(
        description="Analyze GLTF mesh physical properties."
    )
    parser.add_argument("gltf_path", help="Path to the GLTF mesh file.")
    parser.add_argument("system_prompt_path", help="Path to the system prompt file.")
    parser.add_argument(
        "--num-side-views",
        type=int,
        default=8,
        help="Number of equidistant side views to render.",
    )
    parser.add_argument(
        "--env-path", help="Path to a Blender environment file (.blend)."
    )
    parser.add_argument(
        "--metadata",
        help="Optional metadata about the object (e.g., dimensions, category, "
        "description).",
    )

    args = parser.parse_args()

    result = analyze_mesh_physics(
        gltf_path=args.gltf_path,
        system_prompt_path=args.system_prompt_path,
        num_side_views=args.num_side_views,
        env_path=args.env_path,
        metadata=args.metadata,
    )

    logger.info(json.dumps(result, indent=2))
