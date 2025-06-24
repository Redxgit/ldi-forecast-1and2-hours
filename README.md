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
- labelScaler.pkl, scalerLog.pkl, scalerRobust.pkl and scalerStandard.pkl

### Directories
- **data/** - Input datasets and processed data files
- **figs/** - Generated figures and visualizations

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

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Citation

TBA