import hmc

import tensorflow as tf

import numpy as np

import pickle as pkl

import matplotlib.pyplot as plt

x = np.linspace(-0.5, 0.5, 7)

y = np.cos(x)**2 + np.random.normal(size=x.shape, loc=0, scale=0.01)

y_true = np.cos(x)**2

x, y = tf.convert_to_tensor(x, tf.float32), tf.convert_to_tensor(y, tf.float32)

bayesian_nn = hmc.bnn(x, y, units=[16, 16], n_chain=1)

wl = bayesian_nn.initialize()

chain, trace, final_kernel_results = bayesian_nn.run_hmc(initial_config=wl,
                                                         step_size=4e-6,
                                                         num_results=2000,
                                                         num_steps_between_results=120,
                                                         num_leapfrog_steps=120,
                                                         parallel_iterations=10
                                                         )

target_log_probs = trace

with open('chain.pkl', 'wb') as g:
    pkl.dump([chain, trace, final_kernel_results], g)

with open('target_log_prob.pkl', 'wb') as g:
    pkl.dump(target_log_probs, g)

bayesian_nn.plot_neg_log_likelihood(np.negative(target_log_probs))