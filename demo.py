import torch
import yaml
import numpy as np
import os
import datetime

from torch import nn
from random import shuffle

from utilities import data_loader
from utilities import plot_utils

from utilities.mol_utils import add_noise_to_hot
from utilities.utils import make_dir, change_str, use_gpu


class fc_model(nn.Module):
    """
    Fully Connected Neural Network model class for molecular property prediction.
    """
    def __init__(self, len_max_molec1Hot, alphabet_size, 
                 num_of_neurons_layer1, num_of_neurons_layer2, 
                 num_of_neurons_layer3, batch_first=True):
        """
        Initializes the fully connected model.
        Parameters:
        - len_max_molec1Hot (int): Maximum length of one-hot encoded molecule.
        - alphabet_size (int): Size of the alphabet for molecule encoding.
        - num_of_neurons_layer1 (int): Number of neurons in the first layer.
        - num_of_neurons_layer2 (int): Number of neurons in the second layer.
        - num_of_neurons_layer3 (int): Number of neurons in the third layer.
        - batch_first (bool): If True, the first dimension is the batch size.
        Note: there can be more options for num_of_neurons_layerx added in settings.yml. 
        You can also increase/decrease the number of layers in the function here.
        """
        super(fc_model, self).__init__()

        # Reduce dimension upto second last layer of Encoder
        self.encode_1d = nn.Sequential(
            nn.Linear(len_max_molec1Hot, num_of_neurons_layer1),
            nn.ReLU(),
            nn.Linear(num_of_neurons_layer1, num_of_neurons_layer2),
            nn.ReLU(),
            nn.Linear(num_of_neurons_layer2, num_of_neurons_layer3),
            nn.ReLU(),
            nn.Linear(num_of_neurons_layer3, 1)
        )

    def forward(self, x):
        """
        Forward pass through the model (Note: it's implicictely called when model is created)
        Parameters:
        - x (torch.Tensor): Input data tensor.
        Returns:
        torch.Tensor: Output of the model.
        """
        h1 = self.encode_1d(x)
        return h1
    

def train_model(name, model, directory, args,
                upperbound, prop_name, data_train, prop_vals_train, data_test,
                prop_vals_test, lr_enc, num_epochs, batch_size, run_number, scaler):
    """
    Train the model with the given data.
    Parameters:
    - name (str): Name of the model for saving.
    - model (nn.Module): Model to be trained.
    - directory (str): Directory to save outputs.
    - args: Model and training arguments.
    - upperbound (float): Upper bound for adding noise.
    - prop_name (str): Name of the property being predicted.
    - data_train (array): Training data.
    - prop_vals_train (array): Training property values.
    - data_test (array): Testing data.
    - prop_vals_test (array): Testing property values.
    - lr_enc (float): Learning rate for the encoder.
    - num_epochs (int): Number of training epochs.
    - batch_size (int): Size of each training batch.
    - run_number (int): Identifier for the training run.
    - scaler: Scaler object for data normalization.
    """
    # initialize an instance of the model
    optimizer_encoder = torch.optim.Adam(model.parameters(), lr=lr_enc, weight_decay = 1e-5) #l2-regularization

    # reshape for efficient parallelization
    data_train=torch.tensor(data_train, dtype=torch.float, device=args.device)
    data_test=torch.tensor(data_test, dtype=torch.float, device=args.device)
    reshaped_data_train = torch.reshape(data_train,
                                        (data_train.shape[0],
                                         data_train.shape[1]*data_train.shape[2]))
    reshaped_data_test = torch.reshape(data_test,
                                       (data_test.shape[0],
                                        data_test.shape[1]*data_test.shape[2]))

    # add random noise to one-hot encoding
    reshaped_data_test_edit = add_noise_to_hot(reshaped_data_test, upperbound)
    prop_vals_train=torch.tensor(prop_vals_train,
                                 dtype=torch.float, device=args.device)
    prop_vals_test=torch.tensor(prop_vals_test,
                                dtype=torch.float, device=args.device)

    test_loss=[]
    train_loss=[]
    min_loss = float('inf')

    for epoch in range(num_epochs):

        # add stochasticity to the training
        x = [i for i in range(len(reshaped_data_train))]  # random shuffle input
        shuffle(x)
        reshaped_data_train = reshaped_data_train[x]
        prop_vals_train = prop_vals_train[x]
        reshaped_data_train_edit = add_noise_to_hot(reshaped_data_train,
                                            upper_bound=upperbound)

        for batch_iteration in range(int(len(reshaped_data_train_edit)/batch_size)):

            current_smiles_start, current_smiles_stop = \
                batch_iteration * batch_size, (batch_iteration + 1) * batch_size

            # slice data into batches
            curr_mol=reshaped_data_train_edit[current_smiles_start : \
                                              current_smiles_stop]
            curr_prop=prop_vals_train[current_smiles_start : \
                                      current_smiles_stop]
            # feedforward step
            calc_properties = model(curr_mol)
            calc_properties=torch.reshape(calc_properties,[len(calc_properties)])

            # mean-squared error between calculated property and modelled property
            criterion = nn.MSELoss()
            real_loss=criterion(calc_properties, curr_prop)

            loss = torch.clamp(real_loss, min = 0., max = 50000.).double()

            # backpropagation step
            optimizer_encoder.zero_grad()
            loss.backward()
            optimizer_encoder.step()

        # calculate train set
        calc_train_set_property = model(reshaped_data_train_edit)
        calc_train_set_property=torch.reshape(calc_train_set_property,
                                              [len(calc_train_set_property)])
        criterion = nn.MSELoss()
        real_loss_train=criterion(calc_train_set_property, prop_vals_train)
        real_loss_train_num=real_loss_train.detach().cpu().numpy()

        # calculate test set
        calc_test_set_property = model(reshaped_data_test_edit)
        criterion = nn.MSELoss()
        calc_test_set_property=torch.reshape(calc_test_set_property,
                                             [len(calc_test_set_property)])
        real_loss_test=criterion(calc_test_set_property, prop_vals_test)
        real_loss_test_num=real_loss_test.detach().cpu().numpy()


        print('epoch: '+str(epoch)+' - avg loss: '+ \
              str(np.mean(real_loss_train_num))+', testset: '+ \
              str(np.mean(real_loss_test_num)))

        test_loss.append(real_loss_test_num)
        train_loss.append(real_loss_train_num)

        if real_loss_test_num < min_loss:
            min_loss = real_loss_test_num
            calc_train=calc_train_set_property.detach().cpu().numpy()
            calc_test=calc_test_set_property.detach().cpu().numpy()
            real_vals_prop_train=prop_vals_train.detach().cpu().numpy()
            real_vals_prop_test=prop_vals_test.detach().cpu().numpy()
            curr_best_model = model.state_dict()

    plot_utils.prediction_loss(train_loss, test_loss, directory, run_number)
    plot_utils.test_model_after_train(calc_train, real_vals_prop_train,
                                            calc_test,real_vals_prop_test,
                                            directory, run_number, prop_name, scaler)  
    plot_utils.scatter_residuals(calc_train, real_vals_prop_train,
                                 calc_test, real_vals_prop_test,
                                 directory, run_number, prop_name)
    plot_utils.plot_residuals_histogram(calc_train, real_vals_prop_train,
                                        calc_test, real_vals_prop_test,
                                        directory, run_number, prop_name)
    print('Start saving the model')
    torch.save(curr_best_model, name)
    print('Model saved')
        

def load_model(file_name, args, len_max_molec1Hot, model_parameters):
    """
    Load a pre-trained model.
    Parameters:
    - file_name (str): Path to the model file.
    - args: Model and training arguments.
    - len_max_molec1Hot (int): Maximum length of one-hot encoded molecule.
    - model_parameters (dict): Parameters for the model.
    Returns:
    nn.Module: Loaded model.
    """
    model = fc_model(len_max_molec1Hot, **model_parameters).to(device=args.device)
    model.load_state_dict(torch.load(file_name))
    model.eval()
    return model


def train(directory, args, model_parameters, len_max_molec1Hot, upperbound,
          data_train, prop_vals_train, data_test, prop_vals_test, lr_train,
          num_epochs, batch_size, scaler, alphabet_size):
    """
    High-level function to handle the training process.
    Parameters:
    - directory (str): Directory for saving outputs.
    - args: Model and training arguments.
    - model_parameters (dict): Parameters for the model.
    - len_max_molec1Hot (int): Maximum length of one-hot encoded molecule.
    - upperbound (float): Upper bound for adding noise.
    - data_train (array): Training data.
    - prop_vals_train (array): Training property values.
    - data_test (array): Testing data.
    - prop_vals_test (array): Testing property values.
    - lr_train (float): Learning rate for training.
    - num_epochs (int): Number of training epochs.
    - batch_size (int): Size of each training batch.
    - scaler: Scaler object for data normalization.
    - alphabet_size (int): Size of the alphabet for molecule encoding.
    Returns:
        nn.Module: The trained model.
    """
    print("start training")
    existing_files = os.listdir(directory)
    run_number = sum(1 for file in existing_files if file.endswith('.pt')) + 1
    name = change_str(directory)+f'/r{run_number}.pt'

    model = fc_model(len_max_molec1Hot, alphabet_size, **model_parameters).to(device=args.device)
    model.train()

    train_model(name, model, directory, args, upperbound, prop_name,
                data_train, prop_vals_train, data_test, prop_vals_test,
                lr_train, num_epochs, batch_size, run_number, scaler)

    model.load_state_dict(torch.load(name))
    model.eval()
    return model


if __name__ == '__main__':
    # import hyperparameter and training settings from yaml
    print('Start reading data file...')
    settings=yaml.safe_load(open("settings.yml","r"))

    data_parameters = settings['data']
    data_parameters_str = str(data_parameters['data_size'])

    training_parameters = settings['training']
    training_parameters_str = '{}_{}'.format(training_parameters['num_epochs'],
                                             training_parameters['batch_size'])

    model_parameters = settings['model']
    model_parameters_str = "".join([str(n) for n in model_parameters.values()])

    upperbound_tr = settings['upperbound_tr']

    lr_train=settings['lr_train']
    lr_train=float(lr_train)

    data_size = settings['data']['data_size']

    num_mol = data_size
    file_name = settings['data_preprocess']['smiles_file']
    prop_name = file_name[14:-4]

    # data-preprocessing
    data, prop_vals, alphabet, len_max_molec1Hot, largest_molecule_len, scaler = \
        data_loader.preprocess(num_mol, prop_name, file_name)

    data_train, data_test, prop_vals_train, prop_vals_test \
        = data_loader.split_train_test(data, prop_vals, data_size, 0.75)
    
    args = use_gpu()
    num_epochs = settings['training']['num_epochs']
    batch_size=settings['training']['batch_size']

    #prop_name/dataset_size/num_epochs_batch_size/upperbound/lr/num_neurons_in_each_layer
    directory = change_str('results/{}/{}_{}/{}/{}/{}' \
                           .format(prop_name, data_parameters_str,
                                   training_parameters_str,
                                   upperbound_tr,
                                   lr_train,
                                   model_parameters_str))
    
    make_dir(directory)

    model = train(directory, args, model_parameters, len_max_molec1Hot,
                    upperbound_tr, data_train, prop_vals_train, data_test,
                    prop_vals_test, lr_train, num_epochs, batch_size, scaler,
                    alphabet_size=len(alphabet))