"""
Utilities for handling molecular properties and the conversion between
molecular representations.
"""
import re
import pandas as pd
import numpy as np
import sys
sys.path.append('datasets')
import selfies as sf
from rdkit import RDLogger         

from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import MolFromSmiles
from rdkit.Chem import Descriptors
from rdkit.Chem import Draw
from torch import rand
from utilities.utils import make_dir


def smile_to_hot(smile, largest_smile_len, alphabet):
    """
    Convert a single smile string to a one-hot encoding.
    """
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))

    # pad with ' '
    smile += ' ' * (largest_smile_len - len(smile))

    # integer encode input smile
    integer_encoded = [char_to_int[char] for char in smile]

    # one hot-encode input smile
    onehot_encoded = list()
    for value in integer_encoded:
        letter = [0 for _ in range(len(alphabet))]
        letter[value] = 1
        onehot_encoded.append(letter)
    return integer_encoded, np.array(onehot_encoded)


def multiple_smile_to_hot(smiles_list, largest_molecule_len, alphabet):
    """
    Convert a list of smile strings to a one-hot encoding

    Returned shape (num_smiles x len_of_largest_smile x len_smile_encoding)
    """
    hot_list = []
    for smile in smiles_list:
        _, onehot_encoded = smile_to_hot(smile, largest_molecule_len, alphabet)
        hot_list.append(onehot_encoded)
    return np.array(hot_list)


def selfies_to_hot(selfie, largest_selfie_len, alphabet):
    """
    Convert a single selfies string to a one-hot encoding.
    """
    symbol_to_int = dict((c, i) for i, c in enumerate(alphabet))
    # pad with [nop]
    selfie += '[nop]' * (largest_selfie_len - sf.len_selfies(selfie))

    # integer encode
    symbol_list = sf.split_selfies(selfie)
    integer_encoded = [symbol_to_int[symbol] for symbol in symbol_list]
    # one hot-encode the integer encoded selfie
    onehot_encoded = list()
    for index in integer_encoded:
        letter = [0] * len(alphabet)
        letter[index] = 1
        onehot_encoded.append(letter)

    return integer_encoded, np.array(onehot_encoded)


def multiple_selfies_to_hot(selfies_list, largest_molecule_len, alphabet):
    """
    Convert a list of selfies strings to a one-hot encoding
    """

    hot_list = []
    for s in selfies_list:
        _, onehot_encoded = selfies_to_hot(s, largest_molecule_len, alphabet)
        hot_list.append(onehot_encoded)
    return np.array(hot_list)


def add_noise_to_hot(hot, upper_bound): #TODO is there a way to optimize the encodings (e.g., shorten them)
    """
    Replaces all zeroes with a random float in the range [0,upper_bound]
    """
    return hot+upper_bound*rand(hot.shape) 


def draw_mol_to_file(mol_lst, directory):
    """
    Saves the pictorial representation of each molecule in the list to
    file.
    """

    directory = 'dream_results/mol_pics'
    make_dir(directory)
    for smiles in mol_lst:
        mol = MolFromSmiles(smiles)
        Draw.MolToFile(mol,directory+'/'+smiles+'.pdf')


def LC50_from_molecules():
    """
    Extract list of LC50 from CSV file
    """
    data = pd.read_csv('datasets/LC50_filtered.csv')
    LC50s = data['Filtered LC50[-LOG(mol/L)]'].values
    return LC50s


def LC50_from_single_molecule(smiles): #not used but maybe will be useful
    """
    Extract the LC50 value for a single molecule from the CSV file
    """
    # Read the CSV file
    data = pd.read_csv('datasets/LC50_filtered.csv')
    # Match the SMILES representation with the LC50 value
    LC50 = data[data['Filtered SMILES'] == smiles]['Filtered LC50[-LOG(mol/L)]'].values[0]
    
    return LC50