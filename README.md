# MolToxPred
MolToxPred is a machine learning based tool to predict toxicity scores of small molecules


## Prerequisites:

•	Python 3.7+

•	Java JRE 6+

•	`requirements.txt` file contains all the necessary python packages required.

## Usage:

To use the trained models for predictions:

1. Download and unpack the zip file/ Clone the GitHub library
2. Download the model pickle file from this link https://drive.google.com/file/d/1RUFXEKWa79sVaLM0xIlW58W-nzOsL_YB/view?usp=sharing

3. Create an environment with dependencies using `requirements.txt` file

4. Prepare your input file, the molecules should be in SMILES format. For single molecule SMILES can be entered directly, for multiple molecules prepare a .csv file 

5. Run the MolToxPred by `python main.py`

6. The output file is generated as `results.csv` 
  
The `main.py` file will generate descriptors using RDKit and molecular fingerprints using Padelpy for a molecule. `fingerprints_xml.zip` will be parsed to generate the fingerprints,`feature_list.pkl` will perform the feature selection as described in the manucript and output will be individual fingerprint file with selected features in `Padel` folder. Toxicity prediction of the molecule will happen using the trained model `MolToxPred_joblib`and `results.csv` output will be created having toxicity score.

## Datasets:
An example test set that can be used for prediction (in .csv format) is provided in `sample_SMILES`. The larger training and test SMILES dataset & external validation SMILES dataset is provided [here](https://doi.org/10.5281/zenodo.7792389).
