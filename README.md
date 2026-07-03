# LDi Forecast 1 and 2 Hours Ahead

Repository for the paper "Forecasting Local Geomagnetic Disturbances Using Machine Learning: A Multi-Station Neural Network Approach"

## Overview

This repository contains the implementation of machine learning models for forecasting the Local Disturbance Index (LDi) 1-2 hours in advance using a multi-station neural network approach.

## Repository Structure

### Jupyter Notebooks

- **[0] Storms-selection.ipynb** - Storm event selection and identification
- **[1] LDi-cleanup.ipynb** - Data cleaning and preprocessing
- **[2] LDi-storms.ipynb** - Storm data analysis and preparation
- **[3] LDi-metrics-and-plots.ipynb** - Model evaluation metrics and visualization
- **[4] Test-model.ipynb** - Model testing and forecasting for test storms
- **[5] metrics-evaluation.ipynb** - Comprehensive metrics evaluation

### Python Modules

- **[constants.py](constants.py)** - Project constants and configuration parameters
- **[metrics.py](metrics.py)** - Evaluation metrics and scoring functions
- **[pre_processing.py](pre_processing.py)** - Data preprocessing utilities
- **[reldi_modules.py](reldi_modules.py)** - Core neural network modules and architectures
- **[storm_dates.py](storm_dates.py)** - Storm date definitions and utilities
- **[utils.py](utils.py)** - General utility functions

### Model Files

Provided in multiple formats

- **reldi_1to2h_full_model.pt** - Complete trained model
- **reldi_1to2h_jit.pt** - JIT compiled model for inference
- **reldi_1to2h_weights.pt** - Model weights only
- **reldi_1to2h.pt** - Standard model checkpoint

### Data Processing Objects

- labelScaler.pkl, scalerRobust.pkl and scalerStandard.pkl

### Directories

- **data/** - Input datasets and processed data files
- **figs/** - Generated figures and visualizations

Additionally, inside the data folder, the ldi_calculation subfolder includes the H_value, spline and baseline used to calculate the LDi for all the storms depicted in the manuscript for all the stations. Additionally, there are graphs for each storm for each station showcasing the LDi calculation process.

## Requirements

See [requirements.txt](requirements.txt) for the complete list of dependencies.

## Usage

The notebooks are ready to process the data once the repository has been downloaded. Follow the notebooks in numerical order:

1. Start with **[0] Storms-selection.ipynb** for storm event identification
2. Proceed through **[1] LDi-cleanup.ipynb** and **[2] LDi-storms.ipynb** for data preparation
3. Use **[3] LDi-metrics-and-plots.ipynb** for analysis and visualization
4. Run **[4] Test-model.ipynb** to perform forecasting for the test storms
5. Complete the evaluation with **[5] metrics-evaluation.ipynb**

The forecast for test storms will be generated in the **Test-model.ipynb** notebook.

## $LDi$ Methodology

The method estimates the local geomagnetic disturbance by removing, from the measured horizontal magnetic field, both the slowly varying local background field and the regular solar daily variation. The procedure is performed in local time and starts by defining a nighttime reference from the values measured around local midnight, where the regular daily solar variation is expected to be small. These nighttime averages are shown in the diagnostic plots as `Means 90 mins`. A smooth curve through these nighttime reference values, labelled as `Trend` in the plots, is then used to remove the slow trend from the data. This detrending step allows the daily morphology of the regular solar variation to be analysed independently of the local baseline.

For each day, a constrained smooth curve is fitted to the detrended daily profile. This curve represents the expected solar regular variation, hereafter denoted as SR, and is shown in the diagnostic plots as `Sr non corrected`. The fit is constrained so that the variation remains close to zero during the nighttime intervals, consistent with the assumption that a magnetically quiet day should not show significant oscillations around local midnight. The remaining control points, indicated as `P` in the plots, describe the daytime morphology of the daily variation. The goodness of this fit is used as an indicator of geomagnetic quietness: days whose variation is well described by the quiet-day model are selected as quiet days, whereas days with poorer agreement are considered to contain significant disturbance contributions.

The set of quiet days identified over a sufficiently long interval is then used to estimate the slowly varying local baseline, associated with the internal and crustal contribution of the magnetic field and its long-term temporal evolution. This baseline is obtained by averaging quiet-day reference values over selected periods of the year and interpolating them with a smooth function. Thus, both the regular daily variation and the baseline are inferred from the local observations themselves, avoiding the need to rely on global geomagnetic models or external indices.

Finally, the local disturbance index is calculated as the difference between the observed field and the sum of the estimated baseline and solar regular variation. On quiet days, the regular variation is taken from the fit of the same day, labelled as `Sr valid` in the plots. On disturbed days, it is represented by the nearest or otherwise most appropriate quiet-day curve, labelled as `Sr corrected`. The local disturbance index is therefore defined as the residual:

$$
LD_i(t) = H(t) - \left[ B(t) + SR(t) \right],
$$

where $H(t)$ is the observed horizontal magnetic field, $B(t)$ is the local baseline, and $SR(t)$ is the estimated regular solar daily variation. The resulting residual isolates the local geomagnetic perturbation after removing both the slowly varying background field and the regular daily variation.

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Citation

TBA
