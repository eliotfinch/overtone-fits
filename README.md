# overtone-fits

## Running the code

This code relies on the `qnmfits` package, which must be cloned from [`eliotfinch/qnmfits`](https://github.com/eliotfinch/qnmfits) and installed locally (note that this code is different to what is available on pip). See the README at the `qnmfits` repository for more information. Since we will be performing fits with large numbers of overtones, additional data must be downloaded via

```python
import qnmfits
qnmfits.download_cook_data()
```

## Figures in Coleman & Finch (2025)

### Fig. 1

See [`notebooks/even_more_overtones.ipynb`](notebooks/even_more_overtones.ipynb).

### Fig. 2

See [`notebooks/overtone_morphology.ipynb`](notebooks/overtone_morphology.ipynb).

### Fig. 3

See [`notebooks/ringdown_start_time.ipynb`](notebooks/ringdown_start_time.ipynb).

### Fig. 4

See [`notebooks/frequency_perturbations.ipynb`](notebooks/frequency_perturbations.ipynb).

### Fig. 5

See [`notebooks/noise_level.ipynb`](notebooks/noise_level.ipynb).

### Fig. 6

See [`notebooks/amplitude_correlations.ipynb`](notebooks/amplitude_correlations.ipynb).

### Fig. 7

See [`notebooks/frequency_perturbations_bayesian_amplitudes.ipynb`](notebooks/frequency_perturbations_bayesian_amplitudes.ipynb).

### Fig. 8

See [`notebooks/qnm_taxonomy.ipynb`](notebooks/qnm_taxonomy.ipynb).

### Fig. 9

See [`notebooks/ringdown_start_time.ipynb`](notebooks/ringdown_start_time.ipynb).
