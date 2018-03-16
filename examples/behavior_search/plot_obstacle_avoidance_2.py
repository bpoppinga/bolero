"""
==================
Obstacle Avoidance
==================

We use CMA-ES to optimize a DMP so that it avoids point obstacles.
"""
print(__doc__)

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from bolero.environment import External
from bolero.behavior_search import BlackBoxSearch
from bolero.optimizer import CMAESOptimizer
from bolero.representation import DummyBehavior
from bolero.controller import Controller


n_episodes = 500

while not os.path.isfile("parameters_initial.dat"):
    print("waiting...")
    time.sleep(2)
array = np.tile(pd.read_csv("parameters_initial.dat",header=None),(1,1)).flatten()



beh = DummyBehavior()

env = External(array.size)
opt = CMAESOptimizer(variance=100.0 ** 2, random_state=0,initial_params=array)
bs = BlackBoxSearch(beh, opt)
controller = Controller(environment=env, behavior_search=bs,
                        n_episodes=n_episodes, record_inputs=True)

rewards = controller.learn()
