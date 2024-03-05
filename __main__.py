import hmc

import tensorflow as tf

import numpy as np

import pickle as pkl

import matplotlib.pyplot as plt

x = np.linspace(0,20,200)

y = (np.sin(x / 2.2 - np.pi / 4) * 0.98 + np.sin( - x / 2.2 - np.pi / 4) * 0.98) / 2 + np.random.normal(size=x.shape, loc=0, scale=0.2) + 3

y_true = (np.sin(x / 2.2 - np.pi / 4) * 0.98 + np.sin( - x / 2.2 - np.pi / 4) * 0.98) / 2 + 3

x, y = tf.convert_to_tensor(x, tf.float32), tf.convert_to_tensor(y, tf.float32)

y_true = (np.sin(x.numpy() / 2.2 - np.pi / 4) * 0.98 + np.sin( - x.numpy() / 2.2 - np.pi / 4) * 0.98) / 2 + 3

bayesian_nn = hmc.bnn(x, y, units=[7,7,7], n_chain=1)

wl = bayesian_nn.initialize()

print(len(wl))


chain, trace, final_kernel_results = bayesian_nn.run_nou_step_adapt(initial_config=wl,
                                                         step_size=2e-3,
                                                         num_results=20000,
                                                         num_steps_between_results=0,
                                                         max_tree_depth=7,
                                                         parallel_iterations=10,
                                                                    adaptation_rate=0.00015
                                                         )

target_log_probs = trace
bayesian_nn.plot_neg_log_likelihood(np.negative(target_log_probs))

new_step=final_kernel_results.new_step_size
print(new_step)

w=[]
for sample in chain:
    w.append(sample[-1])

chain, trace, final_kernel_results = bayesian_nn.run_hmc(initial_config=w,
                                                         step_size=new_step,
                                                         num_results=1000,
                                                         num_steps_between_results=0,
                                                         num_leapfrog_steps=20,
                                                         parallel_iterations=10
                                                         )



target_log_probs = trace

with open('chain.pkl', 'wb') as g:
    pkl.dump([chain, trace, final_kernel_results], g)

with open('target_log_prob.pkl', 'wb') as g:
    pkl.dump(target_log_probs, g)

bayesian_nn.plot_neg_log_likelihood(np.negative(target_log_probs))

plt.hist(target_log_probs.numpy().flatten(), bins='auto')
plt.show()

model = bayesian_nn.build_network(chain[::2], chain[1::2])

model_var = bayesian_nn.build_network(wl[::2], wl[1::2])


y_pred=model(bayesian_nn.x)

y_pred_var = model_var(bayesian_nn.x)

plt.plot(bayesian_nn.x[:, 0].numpy(), bayesian_nn.y, '.', label='sample')
plt.plot(bayesian_nn.x[:, 0].numpy(), y_pred[::2,0, :, 0].numpy().T, color='black')
plt.plot(bayesian_nn.x[:, 0].numpy(), y_true, label='True')
plt.plot(bayesian_nn.x[:, 0].numpy(), y_pred_var[0, :, 0], label='variational')
plt.legend()
plt.show()
