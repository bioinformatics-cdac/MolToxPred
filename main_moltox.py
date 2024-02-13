#!/usr/bin/env python
# coding: utf-8



import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.get_logger().setLevel(tf.compat.v1.logging.ERROR)

import pandas as pd
import numpy as np
import time
import os
import wget
import zipfile
import joblib
import pickle
import shutil
import rdkit
from rdkit.ML.Descriptors import Descriptors
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import GraphDescriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import rdFingerprintGenerator
from model_tf import create_tf_model




# Input 
user_input= input( "For SMILES as input directly, Press 1 \nFor Multiple SMILES in a file as input Press 2 \n")

flag = False

if user_input=='1':
    query=input("Input your query molecule in SMILES notation in comma separated fashion: ")
    print(query)
    if query==" ":
        print("No SMILES found! Restart")
        exit()
    else:
        mols = query.split(",")
       	df=pd.DataFrame()
        if len(mols)==1:
            mols.append(mols[0])
            flag = True
        df["SMILES"] = mols
elif user_input=='2':
    df_query=input("Enter path of your csv file with molecules in SMILES notation")
    df=pd.read_csv(df_query)

else:
    print('Enter input properly! Restart')
    exit()


# Convert SMILES to Mol objects
mol_list = []
for i in df["SMILES"]:
    mol = Chem.MolFromSmiles(i)
    if mol is None:
        
        continue
    mol_list.append(mol)

# Calculate descriptors
nms = [x[0] for x in Descriptors._descList]
descriptor_calculator = MoleculeDescriptors.MolecularDescriptorCalculator(nms)
descriptors = []

for mol in mol_list:
    if mol is None:
        
        continue 
    descriptor_values = descriptor_calculator.CalcDescriptors(mol)
    descriptors.append(descriptor_values)

    # Optionally write descriptors to a file
    with open('desc.txt', "a") as f:
        f.write(str(descriptor_values) + "\n")

# Create a DataFrame for descriptors
descriptors_df = pd.DataFrame(descriptors)
descriptors_df.columns = nms

# Calculate IPC descriptor
ipc = []
for i in df['SMILES']:
    smi = i 
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        
        continue
    ipc_value = GraphDescriptors.Ipc(mol, avg=True)
    ipc.append(ipc_value)

# Create a DataFrame for IPC
ipc_df = pd.DataFrame(ipc, columns=['IPC'])

# Combine descriptors and IPC into a single DataFrame
descriptors= pd.concat([ipc_df, descriptors_df.drop(['Ipc'], axis=1)], axis=1)



# Replace infinity values (inf) with a large float value (e.g., 1e9)
descriptors.replace([np.inf, -np.inf], 1e9, inplace=True)

# Replace NaN values with 0
descriptors.fillna(0, inplace=True)


import joblib

# Load the scaler
scaler = joblib.load('scaler.pkl')
numerical_columns=joblib.load('numerical_columns.pkl')

# Select the numerical columns from descriptors
descriptors_numerical = descriptors[numerical_columns]

# Scale the numerical columns
scaled_numerical = scaler.transform(descriptors_numerical)

# Replace the scaled values in the descriptors DataFrame
descriptors[numerical_columns] = scaled_numerical



# Morgan Fingerprints
mol_list_new=np.where([x for x in mol_list if str(x) != 'None'])
mol_list_new = list(filter(None, mol_list))
morgan=[]
for i in mol_list_new:
    m=AllChem.GetMorganFingerprintAsBitVect(i,2,useChirality=True)
    morgan.append(m)
vec1=np.array(morgan)
morgan_fp=pd.DataFrame(vec1)
morgan_fp = morgan_fp.add_prefix('MorganFP')

## calculate Fingerprints

## Preparation of fingerprint present in XML file
#wget.download("https://github.com/dataprofessor/padel/raw/main/fingerprints_xml.zip") 
zObject=zipfile.ZipFile("fingerprints_xml.zip", "r")
padel_path='Padel'
if os.path.exists(padel_path):
    shutil.rmtree(padel_path)
os.mkdir(padel_path)
zObject.extractall(path="Padel")




#import glob to read xml file other programming language we have used before, Python
import glob
xml_files = glob.glob(padel_path+"/*Fingerprinter.xml")
xml_files.sort()
xml_files




#create a fingerprint list
FP_list = [
 'AtomPairs2D',
 'EState',
 'CDKextended',
 'CDK',
 'CDKgraphonly',
 'KlekotaRoth',
 'MACCS',
 'PubChem',
 'Substructure']



#create a dictionary
fp = dict(zip(FP_list, xml_files))

# Preparing data in padelpy format
df.to_csv('Padel/molecule.smi', sep='\t', index=False, header=False)


from padelpy import padeldescriptor
start_time = time.time()
padel_fp=[]
for i in fp:
    fingerprint = i
    fingerprint_output_file = ''.join([fingerprint,'_Fingerprint.csv']) ## name of output file
    fingerprint_descriptortypes = fp[fingerprint]
    
    padeldescriptor(mol_dir='Padel/molecule.smi', 
                d_file=fingerprint_output_file, # name of output file'fingerprint.csv'
                #descriptortypes='SubstructureFingerprint.xml', 
                descriptortypes= fingerprint_descriptortypes,
                detectaromaticity=True,
                standardizenitro=True,
                retainorder=True,
                standardizetautomers=True,
                threads=2,
                removesalt=True,
                log=False,
                fingerprints=True)
    #padel_fp.append(fingerprint_output_file)
print("time - {}".format(time.time()-start_time))


all_files=glob.glob("*Fingerprint.csv")
all_files.sort()

all_files





#concatenation of fp
df_from_each_file = (pd.read_csv(f) for f in all_files)
concatenated_df = pd.concat(df_from_each_file, axis=1)
concatenated_df =concatenated_df.drop(['Name'],axis=1)
concatenated_df.drop(concatenated_df.columns[concatenated_df.columns.str.contains('Unnamed')],axis = 1,inplace=True)


combined_df =pd.concat([descriptors,concatenated_df,morgan_fp], axis=1)



combined_df

#unpickling features
with open("feature_list.pkl", "rb") as fp: 
    feature_list = pickle.load(fp)

feature_list
df_model=combined_df[feature_list]
df_model
# Impute missing values in df_model with zeros
df_model= df_model.fillna(0)

## Load Model
trained_model = joblib.load('stacked_model.joblib')

# Prediction
print("The results are here...")
pred = trained_model.predict_proba(df_model)
pred_class1 = pred[:, 1]  # Extract the probabilities for class 1

# Ask the user for a custom prefix
custom_prefix = input("Enter a custom prefix for the output file (without spaces): ")

# Construct the output file name using the custom prefix
output_filename = f"{custom_prefix}_results.csv"

output_df = pd.concat([df, pd.DataFrame(pred_class1, columns=["Toxicity_Score"])], axis=1)

if flag:
    output_df = output_df.iloc[[0]]

print(output_df)

# Load the structural alerts (SMARTS) 
structural_alerts_df = pd.read_csv("tox21_alerts.csv")  
structural_alerts = structural_alerts_df["SMARTS"].dropna().tolist()

#function to check for substructure matches 
def check_substructure_matching(row, alerts):
    if row["Toxicity_Score"] < 0.5:
        return "Not Applicable"
    
    matching_endpoints = []
    mol = Chem.MolFromSmiles(row["SMILES"])
    if mol is not None:
        for alert in alerts:
            pattern = Chem.MolFromSmarts(alert)
            if mol.HasSubstructMatch(pattern):
                matching_endpoint = structural_alerts_df[structural_alerts_df['SMARTS'] == alert]['Endpoint'].values[0]
                matching_endpoints.append(matching_endpoint)
    return ", ".join(set(matching_endpoints)) if matching_endpoints else "Not Applicable"


# Apply the check_substructure_matching function to each row
output_df["Matched_Endpoints"] = output_df.apply(check_substructure_matching, args=(structural_alerts,), axis=1)

#Save the updated DataFrame to a CSV file
output_df.to_csv(output_filename, index=False)










