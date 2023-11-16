"""
Utilities for visualizing training and dreaming results.
"""
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score


def test_model_after_train(calc_train, real_vals_prop_train,
               calc_test, real_vals_prop_test,
               directory, run_number, prop_name, scaler):
    """Scatter plot comparing ground truth data with the modelled data;
    includes both test and training data."""

    # Inverse transform the calculated and real property values
    calc_train = scaler.inverse_transform(calc_train.reshape(-1, 1)).flatten()
    real_vals_prop_train = scaler.inverse_transform(real_vals_prop_train.reshape(-1, 1)).flatten()
    calc_test = scaler.inverse_transform(calc_test.reshape(-1, 1)).flatten()
    real_vals_prop_test = scaler.inverse_transform(real_vals_prop_test.reshape(-1, 1)).flatten()
    
    # Calculate R^2 and RMSE for train and test data
    r2_train = r2_score(real_vals_prop_train, calc_train)
    rmse_train = mean_squared_error(real_vals_prop_train, calc_train, squared=False)

    r2_test = r2_score(real_vals_prop_test, calc_test)
    rmse_test = mean_squared_error(real_vals_prop_test, calc_test, squared=False)

    plt.figure()

    plt.scatter(calc_train, real_vals_prop_train, color='tab:blue', label='Train set', alpha = 0.5, s=3)
    plt.scatter(calc_test, real_vals_prop_test, color='tab:purple', label='Test set', alpha = 0.5, s=3)
    plt.xlabel('Modelled ' + prop_name)
    plt.ylabel('True ' + prop_name)
    plt.title(f'Comparison of Modelled vs. True {prop_name}')
    metrics_text = (f'Train R^2: {r2_train:.2f}, Train RMSE: {rmse_train:.2f}\n'
                    f'Test R^2: {r2_test:.2f}, Test RMSE: {rmse_test:.2f}')
    plt.text(0.95, 0.05, metrics_text, transform=plt.gca().transAxes, horizontalalignment='right',
             verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    plt.legend(loc='best') 
    name = directory + f'/r{run_number}_test_after_training'
    plt.savefig(name)
    plt.close()



def prediction_loss(train_loss, test_loss, directory, run_number):
    """Plot prediction loss during training of model"""
    print(len(train_loss), train_loss)
    print(len(test_loss), test_loss)
    plt.figure()
    plt.plot(train_loss, color='tab:blue', label='Training Loss')
    plt.plot(test_loss, color='tab:purple', label='Test Loss')
    plt.title('Prediction Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    name = directory + f'/r{run_number}_predictionloss_test&train'
    plt.savefig(name)
    plt.close()


def scatter_residuals(calc_train, real_vals_prop_train,
                   calc_test, real_vals_prop_test,
                   directory, run_number, prop_name):
    """Scatter plot comparing residuals with the modelled data;
    includes both test and training data."""
    # Calculate residuals
    residuals_train = real_vals_prop_train - calc_train
    residuals_test = real_vals_prop_test - calc_test

    plt.figure()
    
    # Plot residuals
    plt.scatter(real_vals_prop_train, residuals_train, color='tab:blue', label='Train set Residuals')
    plt.scatter(real_vals_prop_test, residuals_test, color='tab:purple', label='Test set Residuals')
    
    # For residuals, it's useful to have a horizontal line at y=0 to indicate where residuals would be zero
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
    
    plt.xlabel('Modelled ' + prop_name)
    plt.ylabel('Residuals')
    plt.title('Residuals vs. Modelled ' + prop_name)
    plt.legend(loc='best')
    
    name = directory + f'/r{run_number}_scatter_residuals'
    plt.savefig(name)
    plt.close()


def plot_residuals_histogram(calc_train, real_vals_prop_train,
                             calc_test, real_vals_prop_test,
                             directory, run_number, prop_name):
    """Histogram plot of residuals;
    includes both test and training data."""

    # Calculate residuals
    residuals_train = real_vals_prop_train - calc_train
    residuals_test = real_vals_prop_test - calc_test
    plt.figure()

    # Plot histograms
    plt.hist(residuals_train, color='tab:blue', alpha=0.5, label='Train set Residuals', bins=50)
    plt.hist(residuals_test, color='tab:purple', alpha=0.5, label='Test set Residuals', bins=50)

    # It's useful to have a vertical line at x=0 to indicate where residuals would be zero
    plt.axvline(0, color='black', linewidth=0.5, linestyle='--')

    plt.xlabel('Residual Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Residuals for ' + prop_name)
    plt.legend(loc='best')

    name = directory + f'/r{run_number}_residuals_histogram'
    plt.savefig(name)
    plt.close()


def initial_histogram(prop_dream, directory,
                      dataset_name='QM9', prop_name='LC50'):
    """Plot distribution of property values from a given list of values"""

    plt.figure()
    plt.hist(prop_dream, density=True, bins=30)
    plt.ylabel(prop_name + ' - ' + dataset_name)
    name = directory + '/QM9_histogram'
    plt.savefig(name)
    plt.close()


def running_avg_test_loss(avg_test_loss, directory):
    """Plot running average test loss"""

    plt.figure()
    plt.plot(avg_test_loss)
    plt.xlabel('Epochs')
    plt.ylabel('Running average test loss')
    name = directory + '/runningavg_testloss'
    plt.savefig(name)
    plt.close()