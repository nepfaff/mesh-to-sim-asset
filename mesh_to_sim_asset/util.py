import json
import logging
import os
import signal

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from enum import Enum
from io import StringIO
from pathlib import Path

# Set up logger.
logger = logging.getLogger(__name__)


def run_with_logging(func, *args, **kwargs):
    """Run a function and capture/log its stdout and stderr output.

    Args:
        func: The function to run.
        *args: Positional arguments to pass to the function.
        **kwargs: Keyword arguments to pass to the function.

    Returns:
        The result of the function call.
    """
    stdout_capture = StringIO()
    stderr_capture = StringIO()

    try:
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            result = func(*args, **kwargs)

        # Log captured output at info level.
        stdout_content = stdout_capture.getvalue()
        stderr_content = stderr_capture.getvalue()

        if stdout_content.strip():
            logger.info(f"{func.__name__} stdout: {stdout_content.strip()}")
        if stderr_content.strip():
            logger.info(f"{func.__name__} stderr: {stderr_content.strip()}")

        return result

    except Exception as e:
        # Log any captured output before re-raising.
        stdout_content = stdout_capture.getvalue()
        stderr_content = stderr_capture.getvalue()

        if stdout_content.strip():
            logger.error(f"{func.__name__} stdout: {stdout_content.strip()}")
        if stderr_content.strip():
            logger.error(f"{func.__name__} stderr: {stderr_content.strip()}")

        raise


@contextmanager
def comprehensive_error_handling():
    """Context manager to handle all types of errors including segfaults, aborts, etc.

    Note: SIGINT (Ctrl+C) is intentionally not handled to allow normal program termination.

    Yields:
        None: The context manager yields control to the wrapped code.
    """

    def signal_handler(signum, frame):
        signal_names = {
            signal.SIGSEGV: "Segmentation fault",
            signal.SIGABRT: "Abort signal",
            signal.SIGFPE: "Floating point exception",
            signal.SIGILL: "Illegal instruction",
            signal.SIGBUS: "Bus error",
            signal.SIGTERM: "Termination signal",
        }
        signal_name = signal_names.get(signum, f"Signal {signum}")
        raise RuntimeError(f"{signal_name} occurred during mesh processing")

    # Store original handlers.
    original_handlers = {}
    signals_to_handle = [
        signal.SIGSEGV,
        signal.SIGABRT,
        signal.SIGFPE,
        signal.SIGILL,
        signal.SIGBUS,
        signal.SIGTERM,
        # Note: SIGINT is intentionally excluded to allow Ctrl+C termination
    ]

    for sig in signals_to_handle:
        try:
            original_handlers[sig] = signal.signal(sig, signal_handler)
        except (OSError, ValueError):
            # Some signals might not be available on all platforms.
            pass

    try:
        yield
    finally:
        # Restore original signal handlers.
        for sig, handler in original_handlers.items():
            try:
                signal.signal(sig, handler)
            except (OSError, ValueError):
                pass


class CollisionGeomType(Enum):
    """The type of collision geometry to use for the mesh."""

    VTK = "vtk"
    CoACD = "coacd"
    VHACD = "vhacd"


class OpenAIModelType(Enum):
    """The model type to use."""

    GPT4 = "gpt4"
    O3 = "o3"

    def get_model_id(self) -> str:
        return {"GPT4": "gpt-4.1-2025-04-14", "O3": "o3-2025-04-16"}[self.name]


def get_basename_without_extension(path: str) -> str:
    """
    Extracts the basename of a file from a given path, without the extension.

    Args:
        path (str): The full path to the file.

    Returns:
        str: The basename of the file without the extension.
    """
    basename = os.path.basename(path)
    name_without_extension = os.path.splitext(basename)[0]
    return name_without_extension


def find_mesh_files(
    directory: str,
    extensions: list[str] = [
        ".obj",
        ".ply",
        ".fbx",
        ".blend",
        ".dae",
        ".usd",
        ".usda",
        ".usdz",
        ".glb",
        ".gltf",
    ],
) -> list[Path]:
    """Find all mesh files with specified extensions in a directory and its
    subdirectories.

    Args:
        directory (str): Root directory to search in.
        extensions (list[str]): List of file extensions to look for. Defaults to
            [".obj", ".ply", ".fbx", ".blend", ".dae", ".usd", ".usda", ".usdz", ".glb",
            ".gltf"].

    Returns:
        list[Path]: List of paths to found mesh files.
    """
    directory = Path(directory)
    mesh_files = []

    for ext in extensions:
        mesh_files.extend(directory.rglob(f"*{ext}"))

    return mesh_files


def get_metadata_if_exists(mesh_path: Path) -> str | None:
    """Return a string representation of metadata stored alongside a mesh.

    The function looks for a JSON file named `{stem}_metadata.json`
    (e.g. `chair.obj` → `chair_metadata.json`). If it exists, the JSON is loaded
    and pretty-printed so that a downstream LLM can consume it directly.

    Args:
        mesh_path (Path): Path to the mesh file.

    Returns:
        str | None: A human- and LLM-readable JSON string, or None if no metadata file
            is found or it cannot be parsed.
    """
    meta_path = mesh_path.with_name(f"{mesh_path.stem}_metadata.json")

    if not meta_path.is_file():
        return None

    try:
        with meta_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    return json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True)
