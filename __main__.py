import hmc

import tensorflow as tf

import numpy as np

import pickle as pkl

import matplotlib.pyplot as plt

x = np.linspace(-0.5, 0.5, 7)

y = np.cos(x)**2 + np.random.normal(size=x.shape, loc=0, scale=0.01)

x, y = tf.convert_to_tensor(x, tf.float32), tf.convert_to_tensor(y, tf.float32)

bayesian_nn = hmc.bnn(x, y, units=[16, 16], n_chain=1)

wl = bayesian_nn.initialize()

chain, trace, final_kernel_results = bayesian_nn.run_hmc(initial_config=wl,
                                                         step_size=1e-8,
                                                         num_results=40,
                                                         num_burnin_steps=0,
                                                         num_leapfrog_steps=1000000,
                                                         parallel_iterations=10,
                                                         num_steps_between_results=0
                                                         )


with open('chain.pkl', 'wb') as g:
    pkl.dump([chain, trace, final_kernel_results], g)

