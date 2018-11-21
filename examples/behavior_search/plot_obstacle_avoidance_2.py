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
import time

n_episodes = 5000

beh = DummyBehavior()

array = External.get_initial(True)
env = External(array.size,real=True)

opt = CMAESOptimizer(variance=0.000001 ** 2, random_state=0,initial_params=array)
bs = BlackBoxSearch(beh, opt)
controller = Controller(environment=env, behavior_search=bs,
                        n_episodes=n_episodes, record_inputs=True)

rewards = controller.learn()
