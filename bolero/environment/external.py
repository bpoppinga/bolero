# Authors: Bernd Poppinga <poppinga@uni-bremen.de>

import numpy as np
from scipy.spatial.distance import cdist
from .environment import Environment
from bolero.utils.log import get_logger
import os
import pandas as pd
import time
    
class External(Environment):
    """Optimize a trajectory according to some criteria.

    Parameters
    ----------
    x0 : array-like, shape = (n_task_dims,), optional (default: [0, 0])
        Initial position.

    """
    def __init__(self, numInputs, log_to_file=False, log_to_stdout=False,real=False):
        self.numInputs = numInputs
        self.log_to_file = log_to_file
        self.log_to_stdout = log_to_stdout
        self.counter = 0
        self.real = real
        self.dir = "real" if real else "sim"

    @staticmethod
    def get_initial(real):
        dir_ = "real" if real else "sim"
        while not os.path.isfile("/mnt/hgfs/Exchange/"+dir_+"/parameters_0.dat"):
                print("waiting for initial ...")
                time.sleep(2)
        return np.tile(pd.read_csv("/mnt/hgfs/Exchange/"+dir_+"/parameters_0.dat",header=None),(1,1)).flatten()

    def init(self):
        """Initialize environment."""

    def reset(self):
        """Reset state of the environment."""
        self.done = False
        self.counter += 1

    def get_num_inputs(self):
        """Get number of environment inputs.

        Returns
        ----------
        n : int
            number of environment inputs
        """
        return self.numInputs

    def get_num_outputs(self):
        """Get number of environment outputs.

        Returns
        ----------
        n : int
            number of environment outputs
        """
        return self.numInputs

    def get_outputs(self, values):
        """Get environment outputs.

        Parameters
        ----------
        values : array
            Outputs of the environment: positions, velocities and accelerations
            in that order, e.g. for n_task_dims=2 the order would be xxvvaa
        """
        
        # read csv with fitness
        if not os.path.isfile("/mnt/hgfs/Exchange/"+self.dir+"/fitness_"+str(self.counter)+".dat"):
            print("waiting...")
            time.sleep(2)
        else:
            self.done = True
            self.fitness = pd.read_csv("/mnt/hgfs/Exchange/"+self.dir+"/fitness_"+str(self.counter)+".dat",header=None).as_matrix().flatten()
            print(self.fitness)

    def set_inputs(self, values):
        """Set environment inputs, e.g. next action.

        Parameters
        ----------
        values : array,
            Inputs for the environment: positions, velocities and accelerations
            in that order, e.g. for n_task_dims=2 the order would be xxvvaa
        """

        df = pd.DataFrame(values.reshape(1,-1))
        df.to_csv("/mnt/hgfs/Exchange/"+self.dir+"/parameters_"+str(self.counter)+".dat", header=None,index=False)
    

    def step_action(self):
        """Execute step perfectly."""
        #Nothing TODO

    def is_evaluation_done(self):
        """Check if the time is over.

        Returns
        -------
        finished : bool
            Is the episode finished?
        """
        return self.done

    

    def get_feedback(self):
        """Get reward per timestamp based on weighted criteria (penalties)

        Returns
        -------
        rewards : array-like, shape (n_steps,)
            reward for every timestamp; non-positive values
        """
        
        
        
        return self.fitness

    def is_behavior_learning_done(self):
        """Check if the behavior learning is finished.

        Returns
        -------
        finished : bool
            Always false
        """
        return False

    def get_maximum_feedback(self):
        """Returns the maximum sum of feedbacks obtainable."""
        return 10000000000.0 #TODO better value
