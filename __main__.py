import hmc

import tensorflow as tf

import numpy as np

import pickle as pkl

import matplotlib.pyplot as plt

x, y = np.loadtxt('/Users/enricodrigo/Documents/SISSA/bayesian_tf/sportran/examples/data/bayesian/mock_data/mock_data_sin.dat').T

y = (np.sin(x / 2.2 - np.pi / 4) * 0.98 + np.sin( - x / 2.2 - np.pi / 4) * 0.98) / 2 + np.random.normal(size=x.shape, loc=0, scale=0.1)

y_true = (np.sin(x / 2.2 - np.pi / 4) * 0.98 + np.sin( - x / 2.2 - np.pi / 4) * 0.98) / 2

x, y = tf.convert_to_tensor(x[::2], tf.float32), tf.convert_to_tensor(y[::2], tf.float32)

y_true = (np.sin(x.numpy() / 2.2 - np.pi / 4) * 0.98 + np.sin( - x.numpy() / 2.2 - np.pi / 4) * 0.98) / 2

bayesian_nn = hmc.bnn(x, y, units=[10, 10, 10], n_chain=1)

wl = bayesian_nn.initialize()

print(len(wl))


chain, trace, final_kernel_results = bayesian_nn.run_hmc(initial_config=wl,
                                                                step_size=2e-6,
                                                                num_results=1000,
                                                                num_steps_between_results=0,
                                                                num_leapfrog_steps=20,
                                                                parallel_iterations=10
                                                               )


target_log_probs = trace.accepted_results.target_log_prob
bayesian_nn.plot_neg_log_likelihood(np.negative(target_log_probs))


chain, trace, final_kernel_results = bayesian_nn.run_hmc(initial_config=final_kernel_results.proposed_state,
                                                         step_size=1e-6,
                                                         num_results=1000,
                                                         num_steps_between_results=0,
                                                         num_leapfrog_steps=20,
                                                         parallel_iterations=10
                                                         )

target_log_probs = trace.accepted_results.target_log_prob

with open('chain.pkl', 'wb') as g:
    pkl.dump([chain, trace, final_kernel_results], g)

with open('target_log_prob.pkl', 'wb') as g:
    pkl.dump(target_log_probs, g)

bayesian_nn.plot_neg_log_likelihood(np.negative(target_log_probs))

plt.hist(target_log_probs.numpy().flatten(), bins='auto')
plt.show()

model = bayesian_nn.build_network(chain[::2], chain[1::2])

y_pred=model(bayesian_nn.x)

plt.plot(bayesian_nn.x[:, 0].numpy(), bayesian_nn.y, '.', label='sample')
plt.plot(bayesian_nn.x[:, 0].numpy(), y_pred[::2,0, :, 0].numpy().T, color='black')
plt.plot(bayesian_nn.x[:, 0].numpy(), y_true, label='True')
plt.legend()
plt.show()
