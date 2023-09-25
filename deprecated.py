def test_model(directory, args, model, data, data_prop, upperbound):
    """Test model to ensure it is sufficiently trained
       Used to be in test_mol"""

    test_data = torch.tensor(data, dtype=torch.float, device=args.device)
    true_prop = torch.tensor(data_prop, device=args.device)

    # reshape for efficient parallelization
    test_data = test_data.reshape(test_data.shape[0],
                                  test_data.shape[1] * test_data.shape[2])

    # add random noise to one-hot encoding with specified upperbound
    test_data_edit = add_noise_to_hot(test_data, upperbound)

    # feedforward step
    trained_prop = model(test_data_edit)
    trained_prop = trained_prop.reshape(data.shape[0]).clone().detach().numpy()

    # compare ground truth data to modelled data
    plot_utils.test_model_before_dream(trained_prop, true_prop,
                                       directory)
    


def test_model_before_dream(trained_data_prop, computed_data_prop,
                            directory, prop_name='LC50'):s
    """Scatter plot comparing ground truth data with modelled data
       Used to be in plot_utils"""

    plt.figure()
    plt.scatter(trained_data_prop, computed_data_prop, color='tab:blue')
    plt.xlabel('Modelled '+prop_name)
    plt.ylabel('Computed '+prop_name)
    name = directory + '/test_model_before_dreaming'
    plt.savefig(name)
    plt.show()
    closefig()