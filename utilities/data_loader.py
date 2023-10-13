"""
Methods for reading and preprocessing dataset of SMILES molecular strings.
"""

import selfies as sf
import numpy as np
import pandas as pd
import os
from utilities import utils
from utilities import mol_utils
from rdkit import RDLogger   
import logging      


def write_lengths_to_file(filename, smiles_len, selfies_len):
    """Utility function to write the largest SMILES and SELFIES lengths to a file."""
    with open(filename, "w+") as f:
        f.write(f"{smiles_len}\n")
        f.write(f"{selfies_len}\n")


def read_lengths_from_file(filename):
    """Utility function to read the largest SMILES and SELFIES lengths from a file."""
    with open(filename, "r") as f:
        smiles_len = int(f.readline().strip())
        selfies_len = int(f.readline().strip())
    return smiles_len, selfies_len


def get_largest_string_len(selfies_list, smiles_list, prop_name):
    """Returns the length of the largest SELFIES or SMILES string from a list
    of SMILES. If this dataset has been used already,
    then these values will be accessed from a corresponding file."""

    directory = 'dataset_encoding\encoding_info'
    name = os.path.join(directory, f'{prop_name}') 

    if os.path.exists(name):
        os.remove(name)
        #largest_smiles_len, largest_selfies_len = read_lengths_from_file(name) TODO
    #else:
    # Ensure directory exists
    utils.make_dir(directory)

    # Compute the lengths
    largest_smiles_len = len(max(smiles_list, key=len))
    largest_selfies_len = max(sf.len_selfies(s) for s in selfies_list)
    # Store the lengths
    write_lengths_to_file(name, largest_smiles_len, largest_selfies_len)

    return largest_smiles_len, largest_selfies_len


def get_selfies_alphabet(selfies_list):
    """Returns a sorted list of all SELFIES tokens required to build a
    SELFIES string for each molecule."""
    all_selfies_symbols = sf.get_alphabet_from_selfies(selfies_list)
    all_selfies_symbols.add('[nop]')
    all_selfies_symbols.add('.')
    selfies_alphabet = list(all_selfies_symbols)
    selfies_alphabet.sort()
    return selfies_alphabet


def write_alphabet_to_file(prop_name, alphabet):
    """Utility function to write alphabet to a file."""
    
    # Ensure the directory exists
    directory = os.path.dirname(prop_name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    with open(prop_name, "w+") as f:
        f.write('alphabet\n')
        for symbol in alphabet:
            f.write(symbol + '\n')


def read_alphabet_from_file(filename):
    """Utility function to read alphabet from a file."""
    df = pd.read_csv(filename)
    return np.asanyarray(df.alphabet)


def get_string_alphabet(selfies_list, smiles_list, prop_name, filename):
    """Returns a sorted list of all SELFIES tokens and SMILES tokens required
    to build a string representation of each molecule. If this dataset has
    already been used, then these will be accessed from a correspondning file."""

    directory = 'dataset_encoding'
    name1 = os.path.join(directory, f'smiles_alphabet_info\{prop_name}') 
    name2 = os.path.join(directory, f'selfies_alphabet_info\{prop_name}') 

    # Check if the alphabets are already saved, else generate and save them
    if os.path.exists(name1) and os.path.exists(name2):
        os.remove(name1)
        os.remove(name2)
        #smiles_alphabet = read_alphabet_from_file(name1) TODO
        #selfies_alphabet = read_alphabet_from_file(name2)
    #else:
    # Ensure directory exists
    utils.make_dir(directory)

    # Generate SMILES alphabet
    smiles_alphabet = list(set(''.join(smiles_list)))
    smiles_alphabet.append(' ')  # for padding
    smiles_alphabet.sort()
    write_alphabet_to_file(name1, smiles_alphabet)

    # Generate SELFIES alphabet
    selfies_alphabet = get_selfies_alphabet(selfies_list)
    write_alphabet_to_file(name2, selfies_alphabet)

    return smiles_alphabet, selfies_alphabet


def get_selfie_and_smiles_info(selfies_list, smiles_list, prop_name, filename):
    """Returns the length of the largest string representation and the list
    of tokens required to build a string representation of each molecule."""

    largest_smiles_len, largest_selfies_len = get_largest_string_len(selfies_list, smiles_list, prop_name)
    smiles_alphabet, selfies_alphabet = get_string_alphabet(selfies_list, smiles_list, filename, prop_name)
    return selfies_alphabet, largest_selfies_len, smiles_alphabet, largest_smiles_len


def get_selfie_and_smiles_encodings(smiles_list, nrows=0):
    """
    Returns encoding of largest molecule in
    SMILES and SELFIES, given a list of SMILES molecules.
    input:
        - list of SMILES
        - number of rows to be read.
    output:
        - selfies encoding
        - smiles encoding
    """

    if nrows:
        smiles_list = np.random.choice(smiles_list, nrows, replace=False)
    print('--> Translating SMILES to SELFIES...')
    selfies_list = list(map(sf.encoder, smiles_list))

    print('Finished translating SMILES to SELFIES.')

    return selfies_list, smiles_list


def read_data(prop_name, filename):
    """Returns the list of SMILES from a csv file of molecules.
    Column's name must be 'Filtered SMILES'."""

    df = pd.read_csv(filename)
    smiles_list = np.asanyarray(df['SMILES'])
    print('SMILES read', smiles_list[:10])
    prop_list = np.asanyarray(df[prop_name])
    print(f'{prop_name} read', smiles_list[:10])
    return smiles_list, prop_list


def preprocess(num_mol, prop_name, file_name):
    """Takes a random subset of num_mol SMILES from a given dataset;
    converts each SMILES to the SELFIES and creates one-hot encoding;
    encodes other string information."""
    print(f'Loading SMILES and {prop_name} data...')
    smiles_list, prop_list = read_data(prop_name, file_name)
    print('Translating SMILES to SELFIES...')
    selfies_list, smiles_list = get_selfie_and_smiles_encodings(smiles_list, num_mol)
    print('Finished reading SMILES data.\n')
    selfies_alphabet, largest_selfies_len, smiles_alphabet, largest_smiles_len \
        = get_selfie_and_smiles_info(selfies_list, smiles_list, prop_name, file_name) #TODO
    print(f'Loading {prop_name} of all molecules...')
    #RDLogger.DisableLog('rdApp.*')
    #logging.basicConfig(level=logging.INFO) 
    #RDLogger.DisableLog('rdApp.error.explicitValence') TODO

    print('Representation: SELFIES')
    alphabet = selfies_alphabet
    encoding_list = selfies_list
    largest_molecule_len = largest_selfies_len
    print('--> Creating one-hot encoding...')
    data = mol_utils.multiple_selfies_to_hot(encoding_list,
                                             largest_molecule_len,
                                             alphabet)
    print('    Finished creating one-hot encoding.\n')

    len_max_molec = data.shape[1]
    len_alphabet = data.shape[2]
    len_max_molec1Hot = len_max_molec * len_alphabet
    print(' ')
    print('Alphabet has ', len_alphabet, ' letters, largest molecule is ',
          len_max_molec, ' letters.')

    return data, prop_list, alphabet, len_max_molec1Hot, largest_molecule_len


def split_train_test(data, prop_vals, num_mol, frac_train):
    """Split data into training and test data. frac_train is the fraction of
    data used for training. 1-frac_train is the fraction used for testing."""
    data = data[:num_mol]
    prop_vals = prop_vals[:num_mol]

    # Shuffle indices
    indices = np.arange(num_mol)
    np.random.shuffle(indices)

    # Split indices for train and test
    idx_split = int(num_mol * frac_train)
    train_indices = indices[:idx_split]
    test_indices = indices[idx_split:]

    data_train = data[train_indices]
    prop_vals_train = prop_vals[train_indices]
    data_test = data[test_indices]
    prop_vals_test = prop_vals[test_indices]

    return data_train, data_test, prop_vals_train, prop_vals_test