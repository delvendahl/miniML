from functools import cache
from importlib import resources

import h5py

import miniml.resources
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


def get_app_icon_file_path() -> str:
    """
    Returns the path to the application icon file in the resources folder.
    """
    with resources.as_file(
        resources.files(miniml.resources).joinpath("minML_icon.png")
    ) as icon_path:
        return icon_path.as_posix()


def get_hdf_keys(filepath: str) -> list:
    """
    Returns a list of keys in an hdf5 file.
    """
    with h5py.File(filepath, "r") as f:
        return list(f.keys())
