from functools import cache
from importlib import resources

import h5py

import miniml.resources
from miniml.core.trace import MiniTrace
from miniml.fileio.util import is_keras_model


def hex_to_rgb(hexa):
    """
    Convert a hex color code to a tuple of RGB values.
    """
    return tuple(int(hexa[i : i + 2], 16) for i in (1, 3, 5))


def get_available_models() -> list[str]:
    """
    Returns a list of available model paths in the resources/models folder.
    The list only contains relative paths.
    """
    # Look for models inside the package directory
    with resources.as_file(
        resources.files(miniml.resources).joinpath("models")
    ) as models_dir:
        if not models_dir.exists():
            return []
        models = [
            str(p.relative_to(models_dir))
            for p in models_dir.glob("**/*.h5")
            if is_keras_model(str(p))
        ]

    return models


@cache
def get_icon_file_path(icon_name: str) -> str:
    """
    Returns the path to an icon file in the resources/icons folder.
    """
    with resources.as_file(
        resources.files(miniml.resources).joinpath(f"icons/{icon_name}")
    ) as icon_path:
        return icon_path.as_posix()


def get_hdf_keys(filepath: str) -> list:
    """
    Returns a list of keys in an hdf5 file.
    """
    with h5py.File(filepath, "r") as f:
        return list(f.keys())


def load_trace_from_file(file_type: str, file_args: dict) -> MiniTrace:
    """
    Loads a trace from file and returns a MiniTrace object.

    Parameters:
    file_type (str): Type of file to load. Supported types are 'HEKA DAT', 'AXON ABF', and 'HDF5'.
    file_args (dict): Dictionary of arguments to pass to the file loader.

    Returns:
    MiniTrace: MiniTrace object created from the loaded data.

    Raises:
    ValueError: If file_type is not supported.
    """
    file_loader = {
        "HEKA DAT": MiniTrace.from_heka_file,
        "AXON ABF": MiniTrace.from_axon_file,
        "HDF5": MiniTrace.from_h5_file,
    }.get(file_type, None)

    if file_loader is None:
        raise ValueError("Unsupported file type.")

    return file_loader(**file_args)
