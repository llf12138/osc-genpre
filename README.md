# osc-genpre

## Dataset
The data folder contains all the databases.

## Requirements
Required Python packages (for use in a Python 3.9 environment) include:
- `torch==1.12.0`
- `numpy==  2.0.2`
- `pandas== 2.2.2 `
- `scipy==1.13.1`
- `scikit-learn==1.5.2`
- `gplearn==0.4.2`
- `rdkit==2023.3.2 `

## Model
The model folder contains the trained model's .pt file.


## ipynb
- `CNN_Pre.ipynb contains the code for training a CNN to predict PCE.`
- LSTM_Pre.ipynb contains the code for training an LSTM to predict PCE.
- gplearn.ipynb contains the code for training the gplearn model.
- Hyperparameter_T.ipynb contains the code used for hyperparameter tuning of the prediction models.
- gen_pt.ipynb contains the code used for encoding data to be used by the generative model.
- LSTM_Gen.ipynb contains the code for training the LSTM generative model.
- new_Gen.ipynb contains the code for using the trained generative model to generate molecules.

