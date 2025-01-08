# Installation

miniML is written in Python. To use miniML, clone the project's GitHub Repository and install the requirements. 

`git clone https://github.com/delvendahl/miniML.git`

```{hint} 
We recommend creating a virtual environment for miniML using Python version 3.9 or 3.10
```

The Python dependencies for miniML are: 
- sklearn
- matplotlib
- h5py
- pandas
- numpy
- scipy
- tensorflow
- pyabf

To install all dependencies using pip, run the following command in your Python environment:

`pip install -r requirements.txt`

This will install everything you need to run miniML locally.

```{important}
The release of TensorFlow 2.16 and Keras 3 introduced breaking changes that raise an error when loading models trained with earlier TensorFlow versions. To avoid this, it is recommended to use TensorFlow 2.14 or 2.15.
```

miniML can be run on a GPU to speed model inference. 
Either CUDA or tensorflow-metal are required for GPU use. Installation instructions for these requirements
may depend on the specific hardware and OS and can be found online.

