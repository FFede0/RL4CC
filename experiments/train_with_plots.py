import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

from RL4CC.experiments.train import TrainingExperiment

class TrainingExperimentWithPlots(TrainingExperiment):
    def run(self):
        super().run()
        # plot
        self.plot()
        return 
    
    def plot(self):
        data_folder = self.logdir

        plots_folder = os.path.join(data_folder, 'plots')
        os.makedirs(plots_folder, exist_ok=True)

        # Load data
        with open(os.path.join(data_folder, 'result.json')) as f:
            result = json.load(f)

        trial_length_time = result['custom_metrics']['current_time'][0][-1]
        timestamps = [x+(sublist_index*trial_length_time) for sublist_index, sublist in enumerate(result['custom_metrics']['current_time']) for x in sublist]
            
        # create a plot for each custom metric
        for key in result['custom_metrics'].keys():
            if key != 'current_time':
                #if values[0][0] is an array, then convert all values to a single array
                if isinstance(result['custom_metrics'][key][0], list):
                    if isinstance(result['custom_metrics'][key][0][0], list):
                        values = np.array(result['custom_metrics'][key]).flatten()
                    elif ((isinstance(result['custom_metrics'][key][0][0], int)) or (isinstance(result['custom_metrics'][key][0][0], float))):
                        values = np.array(result['custom_metrics'][key]).flatten()
                    else:
                        print('Error: unknown type')
                else:
                    print(f"Error: custom metric {key} is not a list of lists")

                plt.plot(timestamps, values, label=key)
                plt.xlabel('time')
                plt.ylabel(key)
                plt.legend()
                plt.title(key)
                plt.savefig(os.path.join(plots_folder, f'{key}.png'))
                plt.close()

        
        
