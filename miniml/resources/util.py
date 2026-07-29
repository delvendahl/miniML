from functools import cache
from importlib import resources

from miniml.fileio.util import is_keras_model


def get_available_models() -> list[str]:
    """
    Returns a list of available model paths in the resources/models folder.
    The list only contains relative paths.
    """
    # Look for models inside the package directory
    with resources.as_file(
        resources.files(str(__package__)).joinpath("models")
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
        resources.files(str(__package__)).joinpath(f"icons/{icon_name}")
    ) as icon_path:
        return icon_path.as_posix()


def get_app_icon_file_path() -> str:
    """
    Returns the path to the application icon file in the resources folder.
    """
    with resources.as_file(
        resources.files(str(__package__)).joinpath("minML_icon.png")
    ) as icon_path:
        return icon_path.as_posix()
