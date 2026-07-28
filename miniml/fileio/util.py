import h5py


def is_keras_model(filepath: str) -> bool:
    """
    Checks if a given HDF5 contains a keras model.
    """
    with h5py.File(filepath, "r") as f:
        return "keras_version" in f.attrs
