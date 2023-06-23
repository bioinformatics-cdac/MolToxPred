#!/usr/bin/env python
# coding: utf-8


# # Installations
# pip install rdkit-pypi
# pip install padelpy


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

if user_input=='1':
    query=input("Input your query molecule in SMILES notation in comma separated fashion: ")
    print(query)
    if query==" ":
        print("No SMILES found! Restart")
        exit()
    else:
        df=pd.DataFrame(query.split(","))
        df.columns=["SMILES"]   
elif user_input=='2':
    df_query=input("Enter path of your csv file with molecules in SMILES notation")
    df=pd.read_csv(df_query)
# elif user_input=='':
#     print('No input found')
else:
    print('Enter input properly! Restart')
    exit()

# In[4]:


#convert into mol
mol_list=[]
for i in df["SMILES"]:
    mol=Chem.MolFromSmiles(i)
    mol_list.append(mol)
nms=[x[0] for x in Descriptors._descList]


#calculate descriptors
calc = MoleculeDescriptors.MolecularDescriptorCalculator(nms)
des = []
for m in mol_list:
   if m is None: continue 
   des.append(calc.CalcDescriptors(m))
   f = open('desc.txt', "a")
   X = calc.CalcDescriptors(m)
   f.write( str(X) + "\n"  )
   f.close ()


descriptors= pd.DataFrame(des)
descriptors.columns=nms

ipc=[]
for i in df['SMILES']:
    smi = i 
    ipc.append(GraphDescriptors.Ipc(Chem.MolFromSmiles(smi),avg=True))
dsc=descriptors.drop(['Ipc'],axis=1)
descriptors=pd.concat([pd.DataFrame(ipc,columns=['IPC']),dsc],axis=1)


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

## Load Model
trained_model = joblib.load('moltox_pred.pkl')

# Define the TensorFlow model
#model_tf = create_tf_model()

# Prediction
print("The results are here...")
pred = trained_model.predict_proba(df_model)
pred_class1 = pred[:, 1]  # Extract the probabilities for class 1

output_df = pd.concat([df, pd.DataFrame(pred_class1)], axis=1)
output_df.columns = ['SMILES', 'Toxicity Score']
print(output_df)
output_df.to_csv("results.csv")






