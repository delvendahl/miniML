<img align="left" width="100" height="100" src="minML_icon.png">

# miniML: A deep learning framework for synaptic event detection

[![minimal Python version](https://img.shields.io/badge/Python%3E%3D-3.9-grey.svg?style=for-the-badge&logo=python&labelColor=3670A0&logoColor=white)](https://www.anaconda.com/download/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)](https://www.tensorflow.org)
[![DOI](https://img.shields.io/badge/DOI-10.1101/2023.11.02.565316-grey.svg?style=for-the-badge&logo=doi&labelColor=green&logoColor=white)](https://www.biorxiv.org/content/10.1101/2023.11.02.565316)


This repository contains the code described in the following publication:  
O'Neill P.S., Baccino-Calace M., Rupprecht P., Friedrich R.W., Müller, M., and Delvendahl, I. 
(2023) Deep learning-based synaptic event detection. _bioRxiv_ ([doi:10.1101/2023.11.02.565316](https://www.biorxiv.org/content/10.1101/2023.11.02.565316))  

### UPDATE INFORMATION

With the latest update, we include all the files required to generate and score the data to train a new model and use the model on your data. For the actual training, please refer to the following links to Kaggle:\
Transfer learning: https://www.kaggle.com/code/philipponeill/miniml-transfer-learning \
Full training: https://www.kaggle.com/code/philipponeill/miniml-full-training \
The dataset: https://www.kaggle.com/datasets/philipponeill/miniml-training-data

We had to make one change that will impact scripts that have been written for the previous miniML version, namely that we split the direction parameter into two separate parameters, event_direction and training_direction (see also miniML documentation). in practice, this only means that you have to change direction to event_direction in your scripts.

Feel free to contact us if you have any questions, either via mail, or by opening an issue here on GitHub (chances are that other people have the same question).

### 🧠 ABOUT

miniML is a deep-learning-based tool to detect synaptic events in 1d timeseries data. It uses a CNN-LSTM network architecture that was trained using a large dataset of synaptic events from cerebellar mossy fiber to granule cell synapses. 

In this repository, we provide pretrained models and Python code to run model inference on recorded data. In addition, an application example (cerebellar granule cell recording) is included.

### 💻 INSTALLATION

To use miniML, clone the GitHub Repositiory and install the requirements. The Python dependencies are: sklearn, matplotlib, h5py, pandas, numpy, scipy, tensorflow, pyabf. To install all dependencies using pip, run the following command in your Python environment:

`pip install -r requirements.txt`

miniML can be run on a GPU to speed model inference. Either CUDA or tensorflow-metal are required for GPU use. Installation instructions for these requirements may depend on the specific hardware and OS and can be found online.

### ⏱ RUNNING MINIML

First, a miniML *MiniTrace* object needs to be created containing 1d timeseries data. Currently, miniML supports direct loading from HEKA .dat files, Axon .abf files as well as HDF .h5 files. The *trace* object features discrete methods for loading from these file formats (e.g., **MiniTrace.from_h5_file()**). Data in other file formats need to be imported as Python objects.

Next, a miniML *EventDetection* object is initiated. Here, one needs to specify a miniML model file to use as well as the *trace* object to operate on. 

Finally, model inference can be run using the **detect_events()** method. This method will run miniML over the given data using the specified model. Runtime will depend on data length.

Detected events are analyzed and descriptive statistics can subsequently be saved in different formats (.pickle, .h5, .csv).


### 💡 EXAMPLE

The folder "example_data/" contains an example recording from a cerebellar mossy fiber to granule cell synapse together with a commented Jupyter Notebook ([tutorial](tutorial.ipynb)) illustrating the use of miniML.



### ✉️ CONTACT
philippsean.oneill@uzh.ch or igor.delvendahl@uzh.ch