

# DeepVol

DeepVol is a deep learning-based model designed for financial time series forecasting. It utilizes a modified WaveNet architecture (PyTorch Lightning-based) to predict future volatilities based on high-frequency data. 

## Table of Contents

- [Project Description](#project-description)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Evaluating the Model](#evaluating-the-model)
- [File Structure](#file-structure)
- [License](#license)

## Usage

### Training the Model

To train the model, use the `main.py` script located in the `src` directory. The script requires several command-line arguments which can be configured in the `args.py` file.

```bash
python DeepVol/src/main.py
```

Notice that the current implementation includes some placeholder functions that are not fully implemented. These functions are:

- **_get_available_tickers**: This function is intended to retrieve a list of available tickers from the dataset. However, it is not fully implemented and might require adjustments to fit specific data sources.
  
- **_setup_dataloaders**: This function is meant to set up data loaders for training, validation, and testing datasets. It currently serves as a placeholder and needs to be completed with appropriate logic for loading and preprocessing the data depending on the datasource used.


## File Structure

The project directory is structured as follows:

```
DeepVol_private-main-3/
│
├── README.md
├── .gitignore
├── DeepVol/
│   ├── .DS_Store
│   ├── LICENSE
│   ├── src/
│       ├── metrics.py
│       ├── deepvol.py
│       ├── args.py
│       ├── main.py
└── __MACOSX/ (to be ignored)
```


