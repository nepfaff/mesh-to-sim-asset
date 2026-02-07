# mesh-to-sim-asset

[![PyPI version](https://img.shields.io/pypi/v/mesh-to-sim-asset)](https://pypi.org/project/mesh-to-sim-asset/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/nepfaff/mesh-to-sim-asset/actions/workflows/lint.yaml/badge.svg)](https://github.com/nepfaff/mesh-to-sim-asset/actions/workflows/lint.yaml)

**mesh-to-sim-asset** converts 3D mesh files (OBJ, PLY, FBX, GLTF, USD, etc.) into
simulation-ready assets for [Drake](https://drake.mit.edu/). It automatically estimates physical properties (mass, friction, material) using vision-language models and generates collision geometries, producing complete SDF assets that are ready to drop into a simulation.

## Installation

This repository uses [uv](https://docs.astral.sh/uv) for dependency management.

1. Install uv:
```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Install the dependencies into `.venv`:
```sh
uv sync
```

3. Activate the virtual env:
```sh
source .venv/bin/activate
```

4. Install wine:
```sh
sudo add-apt-repository ppa:ubuntu-wine/ppa
sudo apt-get update
sudo apt-get install wine
```

5. Install the pre-commit hooks:
```sh
pre-commit install
```

6. Install `git-lfs`:
```sh
git-lfs install
git-lfs pull
```

7. Install blender:
```sh
sudo snap install blender --classic
```

8. Ensure that the `OPENAI_API_KEY` environment variable is set to your OpenAI key.

### [Optional] Install `usd2sdf`

Only required for running `make_asset_drake_compatible.py` with USD inputs.

See [here](https://github.com/gazebosim/gz-usd) for installation instructions.

## Installation via pip

We are hosting a pip wheel on PyPi. Please install using
`pip install mesh-to-sim-asset`.

Note that this will only install the python package (replacing repo cloning +
`uv sync`). You will still need to install the remaining dependencies manually:

| Feature | External dependency | Required? |
|---|---|---|
| Mesh rendering / VLM analysis | [Blender](https://www.blender.org/) | Yes |
| VLM property estimation | `OPENAI_API_KEY` env var | Yes |
| VTK collision mesh simplification | [Wine](https://www.winehq.org/) + RoLoPoly (source repo only) | Optional — only for VTK collision mode |
| USD input conversion | [usd2sdf](https://github.com/gazebosim/gz-usd) | Optional — only for USD inputs |

The pip wheel does **not** include the RoLoPoly binary or the validation scripts
(`scripts/`). To use VTK-based collision mesh simplification, clone the repository
and follow the full installation instructions above.

After installing the pip wheel, the scripts can be run using
`python -m create_drake_asset_from_geometry` and
`python -m make_asset_drake_compatible`.

## Usage

### Geometry/ Mesh to Drake SDF Simulation Asset

Pipeline for converting geometry mesh files into Drake simulation assets with visual
geometries, collision geometries, and physical properties. Non-specified properties
are estimated with a VLM.

The pipeline entrypoint is `create_drake_asset_from_geometry.py`.
Please see that script for argument documentation.

Example:
```sh
python create_drake_asset_from_geometry.py \
    input_dir_path \
    -o output_dir_path \
    -mck
```

### Simulation Asset (USD, URDF, SDF, MJX) to Drake SDF Simulation Asset

Pipeline for converting existing simulation assets (e.g., articulated objects) into
Drake simulation assets with visual geometries, collision geometries, and physical
properties. Non-specified properties are estimated with a VLM.

The pipeline entrypoint is `make_asset_drake_compatible.py`.
Please see that script for argument documentation.

## Validate Static Equilibrium

Simulation assets should at least be statically stable. We can validate this by placing
them onto a flat floor:
```
python scripts/test_mesh_sim.py \
    asset_name.sdf \
    --position "0, 0, 0.1" \
    --rotation "0, 0, 0"
```

Use the `--use_ramp` argument to test rolling down a ramp. We recommend
`--position "0, 0, 0.3"` for the ramp setting.

## Development

### Build a new wheel and publish it to PyPi

Build the wheel:
```sh
uv build
```

Publish it:
```sh
uv publish --token $PYPI_TOKEN
```
where `PYPI_TOKEN` refers to the PyPi API token.
