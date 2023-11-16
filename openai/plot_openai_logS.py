import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

prop_name = 'LC50[-LOG(mol_L)]'

dataset = pd.read_csv(f'OpenAI_{prop_name}.csv')

true_values, guessed_values = dataset[f'true_{prop_name}'], dataset[f'guessed_{prop_name}']

r2 = r2_score(true_values, guessed_values)
rmse = mean_squared_error(true_values, guessed_values, squared=False)

plt.figure()

plt.scatter(guessed_values, true_values, color='tab:blue', alpha = 0.5, s=3)
plt.xlabel('Modelled ' + prop_name)
plt.ylabel('True ' + prop_name)
plt.title(f'Comparison of Modelled vs. True {prop_name}')
metrics_text = (f'R^2: {r2:.2f}, RMSE: {rmse:.2f}')
plt.text(0.95, 0.05, metrics_text, transform=plt.gca().transAxes, horizontalalignment='right',
            verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
name = f'OpenAI_{prop_name}'
plt.savefig(name)
plt.close()