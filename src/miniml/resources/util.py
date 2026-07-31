import sys
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
def get_resource_file_path(relpath: str) -> str:
    """
    Return the path to a resource file in the resources folder.

    Parameters
    ----------
    relpath : str
        Relative path to the target file inside the resources package.
    """
    with resources.as_file(resources.files(str(__package__)).joinpath(relpath)) as path:
        return path.as_posix()


def get_icon_file_path(relpath: str) -> str:
    """
    Return the path to an icon file in the resources/icons folder.

    Parameters
    ----------
    relpath : str
        Relative icon path under the icons directory.
    """
    return get_resource_file_path(f"icons/{relpath}")


def get_app_icon_file_path(best=False) -> str:
    """
    Return the path to the application icon file in the resources folder.

    Parameters
    ----------
    best : bool, optional
        If True, always return the highest-resolution PNG icon.
        If False, return a platform-specific default icon.
    """
    if best:
        return get_icon_file_path("app/app_512px.png")

    if sys.platform == "win32":
        return get_icon_file_path("app/app.ico")
    else:
        return get_icon_file_path("app/app_512px.png")
