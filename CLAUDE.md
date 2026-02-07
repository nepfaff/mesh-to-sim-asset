# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in
this repository.

## Overview

This repository converts 3D mesh files into simulation-ready assets with physics 
properties for use in Drake and other physics simulators. It provides two main 
pipelines:

1. **Geometry/Mesh to Drake SDF**: Converts raw mesh files (OBJ, PLY, FBX, GLTF, 
   etc.) into Drake simulation assets
2. **Simulation Asset to Drake SDF**: Converts existing simulation assets (USD, 
   URDF, SDF) to Drake-compatible format

## Environment Setup

This project uses `uv` for dependency management. Essential setup commands:

```bash
# Install dependencies
uv sync

# Activate virtual environment  
source .venv/bin/activate

# Install pre-commit hooks
pre-commit install

# Install git-lfs and pull assets
git-lfs install
git-lfs pull
```

Required external dependencies:
- Blender (`sudo snap install blender --classic`)
- Wine (for Windows RoLoPoly binary)
- OpenAI API key in `OPENAI_API_KEY` environment variable

## Core Commands

### Main Pipeline Scripts

```bash
# Process mesh files to Drake assets
python create_drake_asset_from_geometry.py input_dir -o output_dir -mck

# As pip module (after wheel installation)
python -m create_drake_asset_from_geometry input_dir -o output_dir -mck

# Convert existing simulation assets
python make_asset_drake_compatible.py input_dir -o output_dir

# As pip module
python -m make_asset_drake_compatible input_dir -o output_dir
```

### Testing and Validation

```bash
# Test static equilibrium
python scripts/test_mesh_sim.py asset_name.sdf \
    --position "0, 0, 0.1" --rotation "0, 0, 0"

# Test rolling down ramp
python scripts/test_mesh_sim.py asset_name.sdf \
    --use_ramp --position "0, 0, 0.3"
```

### Package Management

```bash
# Build wheel
uv build

# Publish to PyPI
uv publish --token $PYPI_TOKEN
```

## Architecture

### Core Pipeline Flow

1. **Mesh Conversion** (`mesh_conversion.py`): Convert various formats (OBJ, PLY, 
   FBX, USD, etc.) to GLTF
2. **Mesh Simplification** (`mesh_simplification.py`): Use RoLoPoly for 
   physics-optimized simplification
3. **Physics Analysis** (`physics.py`): VLM-based analysis using OpenAI models 
   to estimate physical properties
4. **Collision Geometry** (`mesh_to_vtk.py`): Generate VTK tetrahedral meshes and 
   CoACD convex decomposition
5. **SDF Generation** (`sdformat.py`): Create Drake-compatible SDF files with 
   physics properties
6. **Canonicalization** (`canonicalize.py`): Orient meshes in canonical pose 
   with bottom at z=0

### Key Components

- **Physics Properties**: Mass, material properties, friction coefficients 
  estimated via VLM analysis of multi-view renders
- **Material Database** (`materials.yaml`): Predefined material properties 
  (Young's modulus, density, friction)
- **Collision Geometries**: Supports both tetrahedral VTK meshes and convex 
  decomposition via CoACD
- **System Prompts**: VLM prompts stored in `mesh_to_sim_asset/data/` (bundled 
  in pip package)

### Data Flow

```
Input Mesh → GLTF → Blender Renders → VLM Analysis → Physics Properties
                ↓
         Simplified Mesh → Collision Geometry (VTK/CoACD) → SDF Asset
```

### Package Structure

- `mesh_to_sim_asset/`: Main package containing all pipeline modules
- `mesh_to_sim_asset/data/`: System prompt files for VLM analysis (included in 
  pip distribution)
- `create_drake_asset_from_geometry.py`: Main entry point for mesh processing
- `make_asset_drake_compatible.py`: Entry point for asset conversion
- `materials.yaml`: Material property database
- `studio.blend`: Blender environment for rendering
- `RoLoPoly/`: Windows binary for mesh simplification

### Important Implementation Details

- System prompts are loaded via `importlib.resources` for pip compatibility, 
  with fallback to filesystem
- VLM analysis generates multi-view renders using Blender with coordinate frame 
  visualization
- Physics properties are computed using material-specific moduli and 
  hydroelastic compliance
- Asset validation includes static equilibrium testing in Drake simulation
- Supports both metric and non-metric input meshes with automatic scaling

## Development Notes

- All mesh files should have unique names across the pipeline
- The package handles both development (filesystem) and pip installation 
  (resource) scenarios for data files
- VLM analysis requires significant computational resources and OpenAI API 
  credits
- RoLoPoly mesh simplification requires Wine on Linux systems
- Pre-commit hooks ensure code formatting and quality standards

## Code Style Guidelines

- **Line Length**: Respect the 88-character line limit for all code and comments
- **Comment Formatting**: Full line comments must end with a period (.)
- **Code Quality**: Follow existing patterns in the codebase for consistency
