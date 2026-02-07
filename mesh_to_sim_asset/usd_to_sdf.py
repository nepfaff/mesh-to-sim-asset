"""
Utility functions for converting USD files to SDF files. All geometries are converted
to GLTF files to preserve textures. This leads to improved visuals over
https://github.com/gazebosim/gz-usd.

Note that this only converts the visual part but ignores collision geometry, inertia,
etc. If desired, one could use a hybrid approach with
https://github.com/gazebosim/gz-usd to convert all properties.
"""

import json
import logging
import os
import re
import subprocess
import tempfile

from pathlib import Path

from lxml import etree as ET
from manipulation.make_drake_compatible_model import MakeDrakeCompatibleModel

from mesh_to_sim_asset.pose_utils import matrix_to_pose_list, pose_list_to_matrix
from mesh_to_sim_asset.sdformat import format_xml_for_pretty_print
from mesh_to_sim_asset.util import run_with_logging

# Set up logger.
logger = logging.getLogger(__name__)


def _remove_extensions_required_from_all_gltf(gltf_dir: Path) -> None:
    """Remove extensionsRequired from all GLTF files in directory.

    This ensures GLTF files don't require KHR_texture_transform, which
    VTK/Drake doesn't support.

    Args:
        gltf_dir: Directory containing GLTF files.
    """
    for gltf_path in gltf_dir.glob("*.gltf"):
        try:
            with open(gltf_path, "r") as f:
                data = json.load(f)
            if "extensionsRequired" in data:
                del data["extensionsRequired"]
                with open(gltf_path, "w") as f:
                    json.dump(data, f, indent=2)
                logger.info(f"Removed extensionsRequired from {gltf_path.name}")
        except Exception as e:
            logger.warning(f"Failed to process {gltf_path}: {e}")


def reverse_prismatic_joint_directions(root: ET.Element) -> None:
    """Reverse the axis direction for all prismatic joints.

    Args:
        root: The root XML element containing the SDF structure.
    """
    joints = root.xpath(".//joint[@type='prismatic']")

    for joint in joints:
        axis_xyz_element = joint.find("axis/xyz")
        if axis_xyz_element is not None and axis_xyz_element.text:
            xyz_values = [float(x) for x in axis_xyz_element.text.strip().split()]
            if len(xyz_values) == 3:
                # Negate all xyz values to reverse the direction.
                negated_xyz = [-x if x != 0 else 0 for x in xyz_values]
                axis_xyz_element.text = (
                    f"{negated_xyz[0]} {negated_xyz[1]} {negated_xyz[2]}"
                )


def update_joint_poses_for_removed_link_poses(root: ET.Element) -> None:
    """Update joint poses to compensate for removed link poses.

    Args:
        root: The root XML element containing the SDF structure.
    """
    # First, collect all link poses before removing them.
    link_poses = {}
    links = root.xpath(".//link")

    for link in links:
        link_name = link.get("name")
        pose_element = link.find("pose")

        if pose_element is not None and pose_element.text:
            pose_values = [float(x) for x in pose_element.text.strip().split()]
            if len(pose_values) == 6:
                link_poses[link_name] = pose_values
            else:
                link_poses[link_name] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        else:
            link_poses[link_name] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Now update joint poses.
    joints = root.xpath(".//joint")

    for joint in joints:
        pose_element = joint.find("pose")

        if pose_element is not None:
            # Get current joint pose.
            if pose_element.text:
                joint_pose_values = [
                    float(x) for x in pose_element.text.strip().split()
                ]
                if len(joint_pose_values) != 6:
                    joint_pose_values = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            else:
                joint_pose_values = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

            # Check if pose has relative_to attribute.
            relative_to = pose_element.get("relative_to")

            if relative_to and relative_to in link_poses:
                # Get the relative_to link pose.
                relative_to_pose = link_poses[relative_to]

                # Convert poses to transformation matrices.
                relative_to_T = pose_list_to_matrix(relative_to_pose)
                joint_T = pose_list_to_matrix(joint_pose_values)

                # Compute new joint pose: relative_to_T * joint_T.
                # This maintains the same effective world position when relative_to link
                # becomes identity.
                new_joint_T = relative_to_T @ joint_T

                # Convert back to pose list.
                new_joint_pose = matrix_to_pose_list(new_joint_T)

                # Update the joint pose element (keep relative_to attribute unchanged).
                pose_element.text = (
                    f"{new_joint_pose[0]:.6f} {new_joint_pose[1]:.6f} "
                    f"{new_joint_pose[2]:.6f} {new_joint_pose[3]:.6f} "
                    f"{new_joint_pose[4]:.6f} {new_joint_pose[5]:.6f}"
                )

            else:
                # Handle case where there's no relative_to - assume relative to parent.
                parent_element = joint.find("parent")

                if parent_element is not None:
                    parent_name = parent_element.text

                    # Get parent link pose (default to identity if not found).
                    parent_pose_values = link_poses.get(
                        parent_name, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                    )

                    # Convert poses to transformation matrices.
                    parent_T = pose_list_to_matrix(parent_pose_values)
                    joint_T = pose_list_to_matrix(joint_pose_values)

                    # Compute new joint pose: parent_T * joint_T.
                    new_joint_T = parent_T @ joint_T

                    # Convert back to pose list.
                    new_joint_pose = matrix_to_pose_list(new_joint_T)

                    # Update the joint pose element.
                    pose_element.text = (
                        f"{new_joint_pose[0]:.6f} {new_joint_pose[1]:.6f} "
                        f"{new_joint_pose[2]:.6f} {new_joint_pose[3]:.6f} "
                        f"{new_joint_pose[4]:.6f} {new_joint_pose[5]:.6f}"
                    )


def convert_usd_to_sdf_usd2sdf(usd_path: Path, sdf_path: Path) -> None:
    """Convert a USD file to an SDF file using usd2sdf and Blender for GLTF export.

    This approach uses usd2sdf to preserve the correct structure and poses, then
    uses Blender to export geometries as GLTF to preserve shaders. The mesh URIs
    in the SDF are then replaced with the GLTF files.

    Args:
        usd_path: Path to the USD file.
        sdf_path: Path to the SDF file.
    """
    # Ensure output directory exists.
    sdf_path.parent.mkdir(parents=True, exist_ok=True)

    # Create directory for GLTF exports.
    gltf_dir = sdf_path.parent / f"{sdf_path.stem}_meshes"
    gltf_dir.mkdir(exist_ok=True)

    intermediate_sdf_path = None
    modified_sdf_path = None

    try:
        # Step 1: Use usd2sdf to convert USD to SDF (preserving structure and poses).
        logger.info("Step 1: Converting USD to SDF using usd2sdf...")
        intermediate_sdf_path = sdf_path.parent / f"{sdf_path.stem}_intermediate.sdf"
        convert_usd_to_sdf_with_usd2sdf(usd_path, intermediate_sdf_path)

        if not intermediate_sdf_path.exists():
            raise RuntimeError(
                f"usd2sdf failed to create intermediate SDF file: {intermediate_sdf_path}"
            )

        logger.info(f"✓ Intermediate SDF created: {intermediate_sdf_path}")

        # Step 2: Use Blender to export all geometries as GLTF.
        logger.info("Step 2: Exporting geometries as GLTF using Blender...")
        gltf_mapping = export_geometries_as_gltf_with_blender(usd_path, gltf_dir)

        if not gltf_mapping:
            logger.warning(
                "Warning: No GLTF files were exported. Using original SDF without GLTF "
                "replacement."
            )
            # Just clean and make Drake-compatible without GLTF replacement.
            clean_and_fix_sdf(intermediate_sdf_path)

            # Make Drake-compatible.
            run_with_logging(
                MakeDrakeCompatibleModel,
                input_filename=intermediate_sdf_path.as_posix(),
                output_filename=sdf_path.as_posix(),
                overwrite=True,
            )
        else:
            logger.info(f"✓ Exported {len(gltf_mapping)} GLTF files")

            # Step 3: Replace mesh URIs in the SDF with GLTF files.
            logger.info("Step 3: Replacing mesh URIs with GLTF files...")
            modified_sdf_path = sdf_path.parent / f"{sdf_path.stem}_modified.sdf"
            replace_mesh_uris_in_sdf(
                intermediate_sdf_path, modified_sdf_path, gltf_mapping
            )

            if not modified_sdf_path.exists():
                raise RuntimeError(
                    f"Failed to create modified SDF file: {modified_sdf_path}"
                )

            logger.info(f"✓ Modified SDF created: {modified_sdf_path}")

            # Step 4: Clean and fix the SDF.
            logger.info("Step 4: Cleaning and fixing SDF...")
            clean_and_fix_sdf(modified_sdf_path)
            logger.info("✓ SDF cleaned and fixed")

            # Step 5: Make Drake-compatible.
            logger.info("Step 5: Making SDF Drake-compatible...")
            run_with_logging(
                MakeDrakeCompatibleModel,
                input_filename=modified_sdf_path.as_posix(),
                output_filename=sdf_path.as_posix(),
                overwrite=True,
            )

        if not sdf_path.exists():
            raise RuntimeError(f"Failed to create final SDF file: {sdf_path}")

        logger.info(f"✓ Final SDF created: {sdf_path}")

    except Exception as e:
        logger.error(f"Error during USD to SDF conversion: {e}")
        # Try to create a basic SDF from the intermediate file if it exists.
        if intermediate_sdf_path and intermediate_sdf_path.exists():
            logger.info("Attempting to create basic SDF from intermediate file...")
            try:
                clean_and_fix_sdf(intermediate_sdf_path)
                run_with_logging(
                    MakeDrakeCompatibleModel,
                    input_filename=intermediate_sdf_path.as_posix(),
                    output_filename=sdf_path.as_posix(),
                    overwrite=True,
                )
                logger.info(f"✓ Basic SDF created: {sdf_path}")
            except Exception as e2:
                logger.error(f"Failed to create basic SDF: {e2}")
                raise e
        else:
            raise e

    finally:
        # Clean up intermediate files.
        if intermediate_sdf_path and intermediate_sdf_path.exists():
            intermediate_sdf_path.unlink()
        if modified_sdf_path and modified_sdf_path.exists():
            modified_sdf_path.unlink()

    logger.info(f"✓ USD to SDF conversion completed successfully: {sdf_path}")


def convert_usd_to_sdf_with_usd2sdf(usd_path: Path, output_path: Path) -> None:
    """Convert USD to SDF using usd2sdf utility.

    Args:
        usd_path: Path to the USD file.
        output_path: Path to the output SDF file.

    Raises:
        RuntimeError: If usd2sdf conversion fails or usd2sdf is not found.
    """
    try:
        # Resolve to absolute paths since usd2sdf needs absolute paths to work properly.
        usd_abs = usd_path.resolve()
        output_abs = output_path.resolve()
        result = subprocess.run(
            ["usd2sdf", str(usd_abs), str(output_abs), "false"],
            check=True,
            encoding="utf-8",
            cwd=usd_abs.parent,
            capture_output=True,
        )

        # Log the output at info level.
        if result.stdout:
            logger.info(f"usd2sdf stdout: {result.stdout}")
        if result.stderr:
            logger.info(f"usd2sdf stderr: {result.stderr}")

    except subprocess.CalledProcessError as e:
        # Log error output before raising.
        if hasattr(e, "stdout") and e.stdout:
            logger.error(f"usd2sdf stdout: {e.stdout}")
        if hasattr(e, "stderr") and e.stderr:
            logger.error(f"usd2sdf stderr: {e.stderr}")
        raise RuntimeError(f"Failed to convert {usd_path} with usd2sdf: {e}")
    except FileNotFoundError:
        raise RuntimeError(
            "usd2sdf not found. Please install it following the instructions in README."
        )


def export_geometries_as_gltf_with_blender(
    usd_path: Path, gltf_dir: Path
) -> dict[str, str]:
    """Export all geometries from USD as GLTF files using Blender.

    Args:
        usd_path: Path to the USD file.
        gltf_dir: Directory to export GLTF files.

    Returns:
        Dictionary mapping original mesh names to GLTF file paths.

    Raises:
        FileNotFoundError: If USD file not found.
        ValueError: If USD file is invalid.
        RuntimeError: If Blender export fails.
    """
    # Validate USD file before processing.
    if not usd_path.exists():
        raise FileNotFoundError(f"USD file not found: {usd_path}")
    if not usd_path.is_file():
        raise ValueError(f"USD path is not a file: {usd_path}")
    if usd_path.stat().st_size == 0:
        raise ValueError(f"USD file is empty: {usd_path}")

    blender_script = f"""
import bpy
import json
import bmesh
import mathutils
from pathlib import Path

USD_FILE = Path('{usd_path.as_posix()}')
OUT_DIR = Path('{gltf_dir.as_posix()}')

# Clear existing mesh objects.
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# Import USD file.
bpy.ops.wm.usd_import(
    filepath=str(USD_FILE),
    import_materials=True,
    import_usd_preview=True,
    set_frame_range=False,
)

OUT_DIR.mkdir(parents=True, exist_ok=True)

def sanitize_filename(name: str) -> str:
    \"\"\"Keep only safe filename characters.\"\"\"
    return "".join(c for c in name if c.isalnum() or c in "_-")

def remove_extensions_required_from_gltf(gltf_path: str):
    \"\"\"Remove extensionsRequired field from GLTF file to make extensions optional.\"\"\"
    try:
        import json
        from pathlib import Path
        
        gltf_file = Path(gltf_path)
        if not gltf_file.exists():
            print(f"Warning: GLTF file not found: {{gltf_path}}")
            return
            
        # Read the GLTF JSON.
        with open(gltf_file, 'r') as f:
            gltf_data = json.load(f)
        
        # Remove extensionsRequired if it exists.
        if 'extensionsRequired' in gltf_data:
            removed_extensions = gltf_data['extensionsRequired']
            del gltf_data['extensionsRequired']
            print(f"✓ Removed extensionsRequired: {{removed_extensions}} from {{gltf_file.name}}")
        
        # Write back the modified GLTF.
        with open(gltf_file, 'w') as f:
            json.dump(gltf_data, f, indent=2)
            
    except Exception as e:
        print(f"Error removing extensionsRequired from {{gltf_path}}: {{e}}")

def get_mesh_identifier(obj):
    \"\"\"Get a unique identifier for the mesh that can be matched with SDF URIs.\"\"\"
    # Try to get the mesh name from the object's data.
    if obj.data and obj.data.name:
        return obj.data.name
    # Fallback to object name.
    return obj.name

def gather_all_mesh_objects():
    \"\"\"Gather all mesh objects in the scene.\"\"\"
    mesh_objects = []
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH' and obj.data:
            mesh_objects.append(obj)
    return mesh_objects

# Get all mesh objects.
mesh_objects = gather_all_mesh_objects()
print(f"Found {{len(mesh_objects)}} mesh objects")

# Export each mesh as a separate GLTF file in local coordinates.
gltf_mapping = {{}}

for mesh_obj in mesh_objects:
    # Get mesh identifier for matching with SDF.
    mesh_id = get_mesh_identifier(mesh_obj)
    
    # Create a duplicate of the object to work with.
    bpy.ops.object.select_all(action='DESELECT')
    mesh_obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_obj
    
    # Duplicate the object.
    bpy.ops.object.duplicate()
    duplicate_obj = bpy.context.active_object
    
    # Clear parent and bake transforms into the mesh.
    bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    
    # Generate GLTF filename.
    safe_name = sanitize_filename(mesh_id)
    gltf_filename = f"{{safe_name}}.gltf"
    gltf_path = OUT_DIR / gltf_filename
    
    # Select only the duplicate for export.
    bpy.ops.object.select_all(action='DESELECT')
    duplicate_obj.select_set(True)
    bpy.context.view_layer.objects.active = duplicate_obj
    
    # Export as GLTF.
    bpy.ops.export_scene.gltf(
        filepath=str(gltf_path),
        export_format='GLTF_SEPARATE',
        use_selection=True,
        export_yup=True,
        check_existing=False,
        export_image_format='AUTO',         # PNG for alpha maps, JPG otherwise.
    )
    
    # Remove extensionsRequired field to avoid KHR_texture_transform being required.
    remove_extensions_required_from_gltf(str(gltf_path))
    
    # Store the mapping with various name variations for better matching.
    gltf_rel_path = gltf_path.relative_to(OUT_DIR.parent).as_posix()
    gltf_mapping[mesh_id] = gltf_rel_path
    gltf_mapping[mesh_obj.name] = gltf_rel_path
    
    # Also store with common variations.
    gltf_mapping[mesh_id.replace("_", "")] = gltf_rel_path
    gltf_mapping[mesh_id.replace("-", "")] = gltf_rel_path
    gltf_mapping[mesh_obj.name.replace("_", "")] = gltf_rel_path
    gltf_mapping[mesh_obj.name.replace("-", "")] = gltf_rel_path
    
    print(f"Exported {{mesh_obj.name}} (mesh: {{mesh_id}}) -> {{gltf_filename}}")
    
    # Clean up duplicate.
    bpy.ops.object.delete()

# Save the mapping.
mapping_path = OUT_DIR / "gltf_mapping.json"
with open(mapping_path, "w") as f:
    json.dump(gltf_mapping, f, indent=2)

print(f"Mapping saved with {{len(gltf_mapping)}} entries")
"""

    # Write script to temporary file.
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(blender_script)
        script_path = f.name

    logger.info(f"Running Blender to export geometries as GLTF...")

    try:
        # Run Blender with the script.
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
                f"Blender GLTF export failed:\n{result.stderr}\n{result.stdout}"
            )

        # Post-process: Remove extensionsRequired from all GLTF files.
        # This runs in the main Python process to ensure it works reliably.
        _remove_extensions_required_from_all_gltf(gltf_dir)

        # Read the GLTF mapping.
        mapping_file = gltf_dir / "gltf_mapping.json"
        if mapping_file.exists():
            with open(mapping_file, "r") as f:
                gltf_mapping = json.load(f)
            # Clean up temporary file.
            mapping_file.unlink()
            return gltf_mapping
        else:
            raise RuntimeError("Failed to create GLTF mapping file")

    finally:
        # Clean up temporary script.
        os.remove(script_path)


def replace_mesh_uris_in_sdf(
    input_sdf_path: Path, output_sdf_path: Path, gltf_mapping: dict[str, str]
) -> None:
    """Replace mesh URIs in SDF with GLTF files based on the mapping.

    Args:
        input_sdf_path: Path to the input SDF file.
        output_sdf_path: Path to the output SDF file.
        gltf_mapping: Dictionary mapping mesh names to GLTF paths.
    """
    # Parse the SDF file.
    root = parse_sdf_file(input_sdf_path)

    # Find all mesh elements and replace their URIs.
    mesh_elements = root.xpath(".//mesh/uri")
    replaced_count = 0

    logger.info(f"Found {len(mesh_elements)} mesh URI elements in SDF")
    logger.info(f"Available GLTF mappings: {list(gltf_mapping.keys())}")

    for uri_element in mesh_elements:
        original_uri = uri_element.text
        if original_uri:
            logger.info(f"Processing URI: {original_uri}")

            # Extract mesh name from URI (handle various formats).
            mesh_name = extract_mesh_name_from_uri(original_uri)
            logger.info(f"Extracted mesh name: {mesh_name}")

            # Look for matching GLTF file using multiple strategies.
            matching_gltf = find_matching_gltf(mesh_name, gltf_mapping)

            if matching_gltf:
                uri_element.text = matching_gltf
                replaced_count += 1
                logger.info(f"✓ Replaced {original_uri} -> {matching_gltf}")
            else:
                logger.warning(
                    f"✗ Warning: No GLTF mapping found for mesh: {mesh_name}"
                )

    logger.info(
        f"Replaced {replaced_count} out of {len(mesh_elements)} mesh URIs with GLTF "
        "files"
    )

    # Write the modified SDF.
    format_xml_for_pretty_print(root)
    ET.ElementTree(root).write(output_sdf_path, pretty_print=True, encoding="utf-8")


def find_matching_gltf(mesh_name: str, gltf_mapping: dict[str, str]) -> str | None:
    """Find the best matching GLTF file for a given mesh name.

    Args:
        mesh_name: The mesh name to match.
        gltf_mapping: Dictionary mapping mesh names to GLTF paths.

    Returns:
        The matching GLTF path or None if no match is found.
    """
    # Strategy 1: Exact match.
    if mesh_name in gltf_mapping:
        return gltf_mapping[mesh_name]

    # Strategy 2: Case-insensitive exact match.
    for key, value in gltf_mapping.items():
        if key.lower() == mesh_name.lower():
            return value

    # Strategy 3: Substring match (both directions).
    for key, value in gltf_mapping.items():
        if mesh_name in key or key in mesh_name:
            return value

    # Strategy 4: Fuzzy matching with common variations.
    mesh_variations = [
        mesh_name,
        mesh_name.replace("_", ""),
        mesh_name.replace("-", ""),
        mesh_name.replace(" ", ""),
        mesh_name.lower(),
        mesh_name.upper(),
    ]

    for variation in mesh_variations:
        for key, value in gltf_mapping.items():
            key_variations = [
                key,
                key.replace("_", ""),
                key.replace("-", ""),
                key.replace(" ", ""),
                key.lower(),
                key.upper(),
            ]

            if variation in key_variations:
                return value

    # Strategy 5: Partial match with common mesh naming patterns.
    # Remove common prefixes and suffixes.
    clean_mesh_name = mesh_name
    for prefix in ["mesh_", "geometry_", "visual_", "collision_"]:
        if clean_mesh_name.startswith(prefix):
            clean_mesh_name = clean_mesh_name[len(prefix) :]

    for suffix in ["_mesh", "_geometry", "_visual", "_collision"]:
        if clean_mesh_name.endswith(suffix):
            clean_mesh_name = clean_mesh_name[: -len(suffix)]

    # Try with cleaned name.
    for key, value in gltf_mapping.items():
        clean_key = key
        for prefix in ["mesh_", "geometry_", "visual_", "collision_"]:
            if clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix) :]

        for suffix in ["_mesh", "_geometry", "_visual", "_collision"]:
            if clean_key.endswith(suffix):
                clean_key = clean_key[: -len(suffix)]

        if (
            clean_mesh_name == clean_key
            or clean_mesh_name in clean_key
            or clean_key in clean_mesh_name
        ):
            return value

    return None


def extract_mesh_name_from_uri(uri: str) -> str:
    """Extract mesh name from URI.

    Args:
        uri: The URI string containing the mesh path.

    Returns:
        The extracted mesh name.
    """
    # Remove file:// prefix if present.
    uri = uri.replace("file://", "")

    # Get the base filename without extension.
    mesh_name = Path(uri).stem

    # Handle common mesh naming patterns.
    # Remove common prefixes/suffixes.
    mesh_name = re.sub(r"_(mesh|geometry|visual|collision)$", "", mesh_name)
    mesh_name = re.sub(r"^(mesh|geometry|visual|collision)_", "", mesh_name)

    return mesh_name


def parse_sdf_file(sdf_path: Path) -> ET.Element:
    """Parse an SDF file and return the root element.

    Args:
        sdf_path: Path to the SDF file.

    Returns:
        The root XML element of the parsed SDF.

    Raises:
        ValueError: If the SDF file has invalid XML syntax.
    """
    with open(sdf_path, "r") as file:
        sdf_content = file.read()

    # Remove XML comments to avoid matching content inside them.
    sdf_content_no_comments = re.sub(r"<!--.*?-->", "", sdf_content, flags=re.DOTALL)

    # Add Drake namespace declaration to the root element if not present.
    if 'xmlns:drake="drake.mit.edu"' not in sdf_content_no_comments:
        # Find the root element and add the namespace declaration.
        root_pattern = r"(<[^>]+?)(\s*>)"
        if re.search(root_pattern, sdf_content_no_comments):
            sdf_content_no_comments = re.sub(
                root_pattern,
                r'\1 xmlns:drake="drake.mit.edu"\2',
                sdf_content_no_comments,
                count=1,
            )

    # Parse the XML to properly handle SDF structure.
    try:
        root = ET.fromstring(sdf_content_no_comments)
    except ET.XMLSyntaxError as e:
        raise ValueError(f"Invalid SDF XML syntax in {sdf_path}: {e}")

    # Register the drake namespace for serialization.
    ET.register_namespace("drake", "drake.mit.edu")

    return root


def remove_world_fixed_joints(root: ET.Element) -> None:
    """Remove fixed joints that weld bodies to the world.

    Example of joints to remove:

    <joint name="FixedJoint_joint" type="fixed">
        <parent>world</parent>
        <child>some_body</child>
    </joint>

    Args:
        root: The root XML element containing the SDF structure.
    """
    # Find all fixed joints that have "world" as parent.
    world_fixed_joints = root.xpath(".//joint[@type='fixed' and parent='world']")

    for joint in world_fixed_joints:
        parent_element = joint.find("parent")
        if parent_element is not None and parent_element.text == "world":
            joint_name = joint.get("name", "unnamed")
            child_element = joint.find("child")
            child_name = child_element.text if child_element is not None else "unknown"

            logger.info(
                f"Removing world fixed joint: {joint_name} (child: {child_name})"
            )

            # Remove the joint from its parent.
            joint_parent = joint.getparent()
            if joint_parent is not None:
                joint_parent.remove(joint)


def clean_and_fix_sdf(sdf_path: Path) -> None:
    """Clean and fix an SDF file. Overwrites the input file.

    Args:
        sdf_path: Path to the SDF file to clean and fix.
    """
    root = parse_sdf_file(sdf_path)

    # Remove fixed joints that weld bodies to the world.
    remove_world_fixed_joints(root)

    # Remove <static>true</static> elements.
    static_elements = root.xpath(".//static[text()='true']")
    for element in static_elements:
        parent = element.getparent()
        if parent is not None:
            parent.remove(element)

    # Remove elements with nan or -nan values.
    nan_elements = root.xpath(".//*[text()='nan' or text()='-nan']")
    for element in nan_elements:
        parent = element.getparent()
        if parent is not None:
            parent.remove(element)

    # Remove scale elements from mesh geometry since scale is baked into meshes.
    scale_elements = root.xpath(".//mesh/scale")
    for element in scale_elements:
        parent = element.getparent()
        if parent is not None:
            parent.remove(element)

    # Remove pose elements from visual and collision geometry since poses are baked into meshes.
    visual_pose_elements = root.xpath(".//visual/pose")
    for element in visual_pose_elements:
        parent = element.getparent()
        if parent is not None:
            parent.remove(element)

    collision_pose_elements = root.xpath(".//collision/pose")
    for element in collision_pose_elements:
        parent = element.getparent()
        if parent is not None:
            parent.remove(element)

    # Update joint poses to compensate for removed link poses, then remove link poses.
    update_joint_poses_for_removed_link_poses(root)

    # Remove link poses since transformations are baked into meshes.
    link_pose_elements = root.xpath(".//link/pose")
    for element in link_pose_elements:
        parent = element.getparent()
        if parent is not None:
            parent.remove(element)

    # Reverse prismatic joint directions.
    reverse_prismatic_joint_directions(root)

    # Remove attributes that are ignored by Drake.
    ignored_tags = [
        "plugin",
        "scene",
        "atmosphere",
        "physics",
        "wind",
        "gravity",
        "magnetic_field",
        "surface",
        "self_collide",
        "allow_auto_disable",
        "enable_wind",
        "cast_shadows",
        "laser_retro",
        "visibility_flags",
        "transparency",
        "lighting",
        "render_order",
        "shader",
        "double_sided",
        "dissipation",
    ]
    for tag in ignored_tags:
        elements_to_remove = root.xpath(f".//{tag}")
        for element in elements_to_remove:
            parent = element.getparent()
            if parent is not None:
                parent.remove(element)

    format_xml_for_pretty_print(root)
    ET.ElementTree(root).write(sdf_path, pretty_print=True, encoding="utf-8")
