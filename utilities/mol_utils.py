"""
Utilities for handling molecular properties and the conversion between
molecular representations.
"""
import torch
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
    Converts a SMILES string to a one-hot encoded representation.
    Parameters:
    - smile (str): A SMILES string to be encoded.
    - largest_smile_len (int): The maximum length of a SMILES string for padding.
    - alphabet (list of str): The list of unique characters used in SMILES strings.
    Returns:
    tuple: A pair consisting of the integer-encoded SMILES and its one-hot encoded numpy array.
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
    Converts a list of SMILES strings to a one-hot encoding representation for each molecule.
    Parameters:
    - smiles_list (list of str): A list of SMILES strings.
    - largest_molecule_len (int): The length of the largest molecule in the list.
    - alphabet (list of str): The alphabet used for one-hot encoding.
    Returns:
    np.array: A numpy array of one-hot encoded representations with shape (num_smiles, len_of_largest_smile, len_smile_encoding).
    """
    hot_list = []
    for smile in smiles_list:
        _, onehot_encoded = smile_to_hot(smile, largest_molecule_len, alphabet)
        hot_list.append(onehot_encoded)
    return np.array(hot_list)


def selfies_to_hot(selfie, largest_selfie_len, alphabet):
    """
    Converts a SELFIES string to its one-hot encoding representation.
    Parameters:
    - selfie (str): The SELFIES string to be encoded.
    - largest_selfie_len (int): Maximum length of SELFIES strings, used for padding.
    - alphabet (list of str): List of unique characters in the SELFIES alphabet.
    Returns:
    tuple: A pair where the first element is a list of integer encodings of the SELFIES string and 
           the second element is the corresponding one-hot encoded numpy array.
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
    Converts a list of SELFIES strings into a one-hot encoded format based on the specified largest molecule length and alphabet.
    Parameters:
    - selfies_list (list of str): A list of SELFIES strings representing molecules.
    - largest_molecule_len (int): The length of the largest molecule in the list, used for encoding.
    - alphabet (list of str): The alphabet set used for one-hot encoding.
    Returns:
    numpy.ndarray: An array of one-hot encoded representations of the SELFIES strings.
    """
    hot_list = []
    for s in selfies_list:
        _, onehot_encoded = selfies_to_hot(s, largest_molecule_len, alphabet)
        hot_list.append(onehot_encoded)
    return np.array(hot_list)


def add_noise_to_hot(hot, upper_bound):
    """
    Adds random noise to a one-hot encoded array. Each zero element in the array is replaced with a random float within the range [0, upper_bound]
    Parameters:
    - hot (array): A one-hot encoded array.
    - upper_bound (float): The upper bound for generating random floats to add as noise.
    Returns:
    array: The modified array with added noise.
    """
    return hot+upper_bound*rand(hot.shape) 


def add_noise_to_unflattened(hot, upper_bound): 
    """
    Adds random noise to a tensor by replacing zero elements with random floats in the range [0, upper_bound].
    Parameters:
    - hot (torch.Tensor): The tensor to which noise is added.
    - upper_bound (float): The upper bound for the random noise values.
    Returns:
    torch.Tensor: The tensor with added noise.
    """
    noise = upper_bound * torch.rand(hot.shape).to(hot.device)
    zero_mask = (hot == 0).float()
    noisy_hot = hot + zero_mask * noise
    return noisy_hot


def draw_mol_to_file(mol_lst, directory):
    """
    Generates and saves PDF files of molecular structures for a list of SMILES strings in the specified directory.
    Parameters:
    - mol_lst (list of str): List of SMILES strings representing molecules.
    - directory (str): The directory path where the PDF files will be saved.
    Note:
    This function overwrites the 'directory' parameter with 'dream_results/mol_pics'.
    """
    directory = 'dream_results/mol_pics'
    make_dir(directory)
    for smiles in mol_lst:
        mol = MolFromSmiles(smiles)
        Draw.MolToFile(mol,directory+'/'+smiles+'.pdf')