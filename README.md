<img align="right" width="100" height="100" src="minML_icon.png">

# miniML: A deep learning framework for synaptic event detection

[![minimal Python version](https://img.shields.io/badge/Python%3E%3D-3.9-3670A0.svg?style=flat&logo=python&logoColor=white)](https://www.anaconda.com/download/)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=flat&logo=TensorFlow&logoColor=white)


This repository contains the code described in the following publication:  
O'Neill P.S., Baccino Calace M., Rupprecht P., Friedrich R.W., Müller, M., and Delvendahl, I. 
(2023) Deep learning-based detection of spontaneous synaptic events. _bioRxiv_ XXX ([doi:xxx](https://tbd))  


---


miniML is a deep-learning-based tool to detect synaptic events in 1d timeseries data. In this repository, we provide pretrained models and Python code to run model inference on recorded data. In addition, an application example is included.


To run miniML, clone the GitHub Repositiory and install the requirements
(run `pip install -r requirements.txt`). Python dependencies are: sklearn, matplotlib, h5py, pandas, numpy, scipy, tensorflow, pyabf. 


The folder "Example/" contains an example recording from a MF–GC synapse together with a commented Jupyter Notebook illustrating the use of miniML.



---  
Contact:
philippsean.oneill@uzh.ch or igor.delvendahl@uzh.ch