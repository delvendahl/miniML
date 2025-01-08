# miniML models

Currently, miniML includes the following models:

```{seealso}
Examples of using these models can be found in the [paper](https://doi.org/10.7554/eLife.98485.1).
```

## miniML base model (GC mEPSC model)

The base model trained from scratch using the GC mEPSC dataset. This model is used as a starting point for transfer learning. The labeled training originates from spontaneous mEPSC recordings of cerebellar granule cells in [Delvendahl et al.](https://doi.org/10.1073/pnas.1909675116).

## Transfer learning models

### 1. miniML GC mEPSP model

Transfer learning model for mEPSP detection.

### 2. zebrafish sEPSC model

Transfer learning model for sEPSC detection. Training data was generated using data from [Rupprecht et al.](https://doi.org/10.1016/j.neuron.2018.09.013).

### 3. Drosophila NMJ wt mEPSC model

Transfer learning model for mEPSC detection in Drosophila NMJ two-electrode voltage-clamp recordings. Training data was generated using data from [Baccino-Calace et al.](https://doi.org/10.7554/eLife.71437).

### 4. Drosophila NMJ glurIIa mEPSC model

Transfer learning model for mEPSC detection in Drosophila NMJ two-electrode voltage-clamp recordings of *GluRIIa* mutants. Training data was generated using data from [Baccino-Calace et al.](https://doi.org/10.7554/eLife.71437).

### 5. iGluSnFR optical minis model

Transfer learning model for mEPSC detection in optical mini recordings using *iGluSnFR*. Training data was generated using data from [Agarwal et al.](https://doi.org/10.1038/s41592-023-01863-6).

```{seealso}
Associated training datasets can be found on [Zenodo](https://doi.org/10.5281/zenodo.14507343).
```