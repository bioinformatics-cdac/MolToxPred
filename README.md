# MolToxPred
MolToxPred is a machine learning based tool to predict toxicity scores of small molecules


## Prerequisites:

•	Python 3.7+

•	Java JRE 6+

•	`requirements.txt` file contains all the necessary python packages required.

## Usage:

To use the trained models for predictions:

1. Download and unpack the zip file/ Clone the GitHub library 

2. Set up the required environment `requirements.txt` file

3. Prepare your input file, the molecules should be in SMILES format. For single molecule SMILES can be entered directly, for multiple molecules prepare a csv file 

4. Run the MolToxPred by `python main.py`

5. The output file is generated as `results.csv` 
  
The trained model is provided as `MolToxPred_joblib` and the by feature selection file as `feature_list.pkl` . The `main.py` file will generate descriptors and fingerprint of the molecule, do the feature selection and predict toxicity of the molecule using the trained model. Output will be probability score for toxicity.
