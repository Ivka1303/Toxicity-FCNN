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
import sklearn 
from sklearn.preprocessing import MinMaxScaler
import logging      


def write_lengths_to_file(filename, smiles_len, selfies_len):
    """
    Writes the maximum lengths of SMILES (1st line) and SELFIES (2nd line) strings to a specified file.
    Parameters:
    - filename (str): Path to the output file.
    - smiles_len (int): Maximum length of SMILES strings.
    - selfies_len (int): Maximum length of SELFIES strings.
    """
    with open(filename, "w+") as f:
        f.write(f"{smiles_len}\n")
        f.write(f"{selfies_len}\n")


def read_lengths_from_file(filename):
    """
    Reads and returns the largest lengths of SMILES and SELFIES strings from a file.
    Parameters:
    - filename (str): The name of the file containing the lengths.
    Returns:
    tuple: A pair of integers representing the largest lengths of SMILES and SELFIES strings.
    """
    with open(filename, "r") as f:
        smiles_len = int(f.readline().strip())
        selfies_len = int(f.readline().strip())
    return smiles_len, selfies_len


def get_largest_string_len(selfies_list, smiles_list, prop_name):
    """
    Calculates and returns the lengths of the largest SELFIES and SMILES strings from provided lists. 
    If a file with these values for the specified property already exists, it updates the file.    
    Parameters:
    - selfies_list (list of str): List of SELFIES strings.
    - smiles_list (list of str): List of SMILES strings.
    - prop_name (str): The property name used to create or update the file with string lengths.
    Returns:
    tuple: A pair containing the length of the largest SMILES string and the largest SELFIES string.
    """

    directory = 'dataset_encoding\encoding_info'
    name = os.path.join(directory, f'{prop_name}') 

    if os.path.exists(name):
        os.remove(name)
    utils.make_dir(directory)
    # Compute the lengths
    largest_smiles_len = len(max(smiles_list, key=len))
    largest_selfies_len = max(sf.len_selfies(s) for s in selfies_list)
    # Store the lengths
    write_lengths_to_file(name, largest_smiles_len, largest_selfies_len)
    return largest_smiles_len, largest_selfies_len


def get_selfies_alphabet(selfies_list):
    """
    Extracts and returns a sorted list of unique SELFIES tokens from a given list of SELFIES strings.
    Parameters:
    - selfies_list (list of str): A list of SELFIES strings representing molecules.
    Returns:
    list of str: A sorted list containing unique SELFIES tokens needed for the given SELFIES strings.
    """
    all_selfies_symbols = sf.get_alphabet_from_selfies(selfies_list)
    all_selfies_symbols.add('[nop]')
    all_selfies_symbols.add('.')
    selfies_alphabet = list(all_selfies_symbols)
    selfies_alphabet.sort()
    return selfies_alphabet


def write_alphabet_to_file(prop_name, alphabet):
    """
    This function creates a directory (if it doesn't exist) based on the provided file path (`prop_name`), 
    then writes the 'alphabet' (unique SELFIES tokens) to the file.
    Parameters:
    - prop_name (str): The path of the file where the alphabet is to be written.
    - alphabet (list of str): A list of alphabet symbols to write to the file.
    """
    # Ensure the directory exists
    directory = os.path.dirname(prop_name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    with open(prop_name, "w+") as f:
        f.write('alphabet\n')
        for symbol in alphabet:
            f.write(symbol + '\n')


def read_alphabet_from_file(filename):
    """
    Reads and returns the alphabet data from a specified file.
    Parameters:
    - filename (str): The path to the file containing alphabet data.
    Returns:
    numpy.ndarray: An array of alphabet characters extracted from the file.
    """
    df = pd.read_csv(filename)
    return np.asanyarray(df.alphabet)


def get_string_alphabet(selfies_list, smiles_list, prop_name, filename):
    """
    Generates and returns sorted lists of unique SMILES and SELFIES tokens from the provided lists. 
    If the alphabets for a given property name exist in a specified directory, they are overwritten. 
    The alphabets are also saved to files for future use.
    Parameters:
    - selfies_list (list of str): List of SELFIES strings.
    - smiles_list (list of str): List of SMILES strings.
    - prop_name (str): Property name to label the alphabet files.
    - filename (str): Base filename to use for saving alphabet files.
    Returns:
    tuple: A pair of lists, where the first list contains sorted SMILES tokens and the second list contains SELFIES tokens.
    """
    directory = 'dataset_encoding'
    name1 = os.path.join(directory, f'smiles_alphabet_info\{prop_name}') 
    name2 = os.path.join(directory, f'selfies_alphabet_info\{prop_name}') 
    # Check if there're old alphabets and remove them 
    if os.path.exists(name1) and os.path.exists(name2):
        os.remove(name1)
        os.remove(name2)
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
    """
    Calculates and returns the maximum string length and the alphabet of tokens for SELFIES and SMILES strings.
    Parameters:
    - selfies_list (list of str): A list of SELFIES strings.
    - smiles_list (list of str): A list of SMILES strings.
    - prop_name (str): The name of the property being analyzed.
    - filename (str): The filename where the alphabet data is stored.
    Returns:
    tuple: A tuple containing the alphabet for SELFIES, maximum SELFIES string length, alphabet for SMILES, 
           and maximum SMILES string length.
    """
    largest_smiles_len, largest_selfies_len = get_largest_string_len(selfies_list, smiles_list, prop_name)
    smiles_alphabet, selfies_alphabet = get_string_alphabet(selfies_list, smiles_list, filename, prop_name)
    return selfies_alphabet, largest_selfies_len, smiles_alphabet, largest_smiles_len


def get_selfie_and_smiles_encodings(smiles_list, nrows=0):
    """
    Given a list of SMILES strings representing molecules, this function returns the encodings for 
    nrows SELFIES and SMILES.
    Parameters:
    - smiles_list (list of str): A list containing SMILES strings of molecules.
    - nrows (int, optional): The number of SMILES strings to process. If 0, all entries in `smiles_list` are processed. Defaults to 0.
    Returns:
    tuple: A pair where the first element is a list of SELFIES encodings and the second element is the processed list of SMILES strings.
    """
    if nrows:
        smiles_list = np.random.choice(smiles_list, nrows, replace=False)
    print('--> Translating SMILES to SELFIES...')
    selfies_list = list(map(sf.encoder, smiles_list))
    print('Finished translating SMILES to SELFIES.')
    return selfies_list, smiles_list


def read_data(prop_name, filename):
    """
    Reads SMILES strings and a specified property from a CSV file containing molecular data.
    Parameters:
    - prop_name (str): The name of the property column to be read from the CSV file.
    - filename (str): Path to the CSV file containing molecular data with 'SMILES' column.
    Returns:
    tuple: A pair where the first element is a list of SMILES strings, and the second element is a list of values for the specified property.
    """
    df = pd.read_csv(filename)
    smiles_list = np.asanyarray(df['SMILES'])
    print('SMILES read')
    prop_list = np.asanyarray(df[prop_name])
    print(f'{prop_name} read',)
    return smiles_list, prop_list


def preprocess(num_mol, prop_name, file_name):
    """
    Randomly selects a subset of molecules, converts their SMILES representations to SELFIES, and performs one-hot encoding, 
    and normalizes molecular property specified by 'prop_name' and encodes it.
    Parameters:
    - num_mol (int): Number of molecules to process.
    - prop_name (str): Name of the molecular property to process.
    - file_name (str): Name of the file containing the dataset.
    Returns:
    tuple: A tuple containing the one-hot encoded SELFIES, normalized property values, alphabet used for encoding,
           total size of the one-hot encoding for the largest molecule, length of the largest molecule in SELFIES format, and the scaler object used for normalization.
    """
    print(f'Loading SMILES and {prop_name} data...')
    smiles_list, prop = read_data(prop_name, file_name)
    scaler = MinMaxScaler()
    prop_list = scaler.fit_transform(prop.reshape(-1, 1)).flatten()
    print('Translating SMILES to SELFIES...')
    selfies_list, smiles_list = get_selfie_and_smiles_encodings(smiles_list, num_mol)
    print('Finished reading SMILES data.\n')
    selfies_alphabet, largest_selfies_len, smiles_alphabet, largest_smiles_len \
        = get_selfie_and_smiles_info(selfies_list, smiles_list, prop_name, file_name) #TODO
    print(f'Loading {prop_name} of all molecules...')

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

    return data, prop_list, alphabet, len_max_molec1Hot, largest_molecule_len, scaler


def split_train_test(data, prop_vals, num_mol, frac_train):
    """
    Splits a dataset into training and testing subsets based on a specified fraction.
    Parameters:
    - data (array-like): The dataset to be split.
    - prop_vals (array-like): Corresponding property values of the dataset.
    - num_mol (int): Number of molecules to consider from the dataset.
    - frac_train (float): The fraction of the dataset to be used for training.
    Returns:
    tuple: Contains four elements in the order: training data, testing data, 
    training property values, testing property values.
    """
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