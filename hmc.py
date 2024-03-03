import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow_probability import math as tfm
import numpy as np
from tensorflow_probability.python.internal import test_util
import math
from functools import partial
import matplotlib.pyplot as plt
import time

import pickle as pkl
import copy

class bnn:
    def __init__(self, x, y, units, n_chain):
        self.x = x[:, None]
        self.y = y
        self.units = units
        self.n_chain = n_chain


    def dense(self, X, W, b, activation):

        return activation(tf.matmul(X, W) + b)

    def build_model(self, activation=tf.nn.tanh, optimizer='Adam', loss = 'mse', learning_rate=1e-3):

        kerasl=[tf.keras.layers.Dense(units=self.units[0],
                                      bias_initializer='random_normal',
                                      kernel_initializer='random_normal',
                                      input_shape=(1,),
                                      activation=activation)]
        for l in self.units[0:]:
            kerasl.append(tf.keras.layers.Dense(units=l,
                                      bias_initializer='random_normal',
                                      kernel_initializer='random_normal',
                                      input_shape=(1,),
                                      activation=activation))

        kerasl.append(tf.keras.layers.Dense(units=1,
                                      kernel_initializer='random_normal',
                                      bias_initializer='random_normal',
                                      activation=activation))

        modeltry = keras.Sequential(kerasl)

        if optimizer=='SGD':
            opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)

        elif optimizer=='Adam':
            opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        else:
            raise ValueError('OPTIMIZER NOT IMPLEMENTED YET!')

        if loss == 'off':
            self.loss = self.loss

        elif loss=='mse':
            self.loss = 'mean_squared_error'

        else:
            raise ValueError('LOSS NOT IMPLEMENTED YET!')

        modeltry(self.x)

        modeltry.compile(opt, loss=self.loss)

        return modeltry

    def loss_off(self, y, y_pred):

        ell = 3
        nu = 2

        rho = tf.clip_by_value(y_pred[:], -0.99, 0.99)

        _alpha = 1 / (1 - rho ** 2)
        _beta = rho / (1 - rho ** 2)
        _lambda = 0.5 * ell * nu
        _gamma2 = _alpha ** 2 - _beta ** 2
        _lambda_minus_half = _lambda - 0.5

        # Data is distributed according to a Variance-Gamma distribution with parameters (notation as in Wikipedia):
        # mu = 0; alpha = 1/(1-rho**2); beta = rho/(1-rho**2); lambda = ell*nu/2
        # Its expectation value is ell*nu*rho
        z = y[:] * ell * nu
        absz = tf.math.abs(z)

        log_pdf = _lambda * tf.math.log(_gamma2) +\
                  _lambda_minus_half * tf.math.log(absz) + \
                  tf.math.log(tfp.math.bessel_kve(_lambda_minus_half, _alpha * absz)) -\
                  _alpha * absz +\
                  _beta * z -\
                  0.5 * tf.math.log(np.pi) -\
                  tf.math.log(tf.math.lgamma(_lambda)) -\
                  _lambda_minus_half * tf.math.log(2 * _alpha)

        posterior_model = -tf.math.reduce_sum(log_pdf)
        return posterior_model



    def initialize(self):
        model = self.build_model()

        model(self.x)

        model.summary()

        model.fit(self.x[:, 0], self.y, batch_size=200, epochs=4000)

        w = []

        for l in model.layers:
            w.append(l.get_weights()[0])

            w.append(l.get_weights()[1])

        with open('variationalmodel.pkl', 'wb') as g:
            pkl.dump(w, g)

        wl = []

        for idx, l in enumerate(model.layers):
            ws=l.get_weights()[0].shape

            wl.append(tf.convert_to_tensor(w[int(idx*2)], dtype=tf.float32) + tf.random.normal(shape=((self.n_chain, )+ws), stddev=0.01))

            bs = l.get_weights()[1].shape

            wl.append(tf.convert_to_tensor(w[int(idx*2)+1], dtype=tf.float32) + tf.random.normal(shape=((self.n_chain, )+bs), stddev=0.01))

        return wl



    def build_network(self, weights_list, biases_list, activation=tf.nn.tanh):

        def model(X):
            net = X[:]
            for (weights, biases) in zip(weights_list[:-1], biases_list[:-1]):

                net = self.dense(net, weights, biases[..., None, :], activation)
            # final linear layer
            net = tf.matmul(net, weights_list[-1]) + biases_list[-1][..., None, :]
            return net

        return model

    def trace_fn(self, current_state, results, summary_freq=100):
        #step = results.step
        tf.print(results.accepted_results.target_log_prob)
        #with tf.summary.record_if(tf.equal(step % summary_freq, 0)):

        #    for idx, tensor in enumerate(current_state, 1):
        #        count = str(math.ceil(idx / 2))
        #        name = "weights_" if idx % 2 == 0 else "biases_" + count
        #        tf.summary.histogram(name, tensor, step=tf.cast(step, tf.int64))
        return results

    def trace_fn_bi_nou(self, current_state, results):
        step = results.step
        tf.print(results.inner_results.accepted_results.target_log_prob)

        return results

    def unnormalized_log_prob_off(self, *args):
        ell = 3
        nu = 2

        weights_list = args[::2]
        biases_list = args[1::2]

        modeltry = self.build_network(weights_list, biases_list)


        rho = tf.clip_by_value(modeltry(self.x), -0.98, 0.98)

        _alpha = 1 / (1 - rho ** 2)
        _beta = rho / (1 - rho ** 2)
        _lambda = 0.5 * ell * nu
        _gamma2 = _alpha ** 2 - _beta ** 2
        _lambda_minus_half = _lambda - 0.5

        # Data is distributed according to a Variance-Gamma distribution with parameters (notation as in Wikipedia):
        # mu = 0; alpha = 1/(1-rho**2); beta = rho/(1-rho**2); lambda = ell*nu/2
        # Its expectation value is ell*nu*rho
        z = self.y * ell * nu
        absz = tf.math.abs(z)

        log_pdf = _lambda * tf.math.log(_gamma2) + \
                  _lambda_minus_half * tf.math.log(absz) + \
                  tf.math.log(tfp.math.bessel_kve(_lambda_minus_half, _alpha * absz)) - \
                  _alpha * absz + \
                  _beta * z - \
                  0.5 * tf.math.log(np.pi) - \
                  tf.math.log(tf.math.lgamma(_lambda)) - \
                  _lambda_minus_half * tf.math.log(2 * _alpha)

        posterior_model = tf.math.reduce_mean(log_pdf, axis=(-1, -2))
        #tf.print(posterior_model)

        return posterior_model

    def unnormalized_log_prob_normal(self, *args):

        weights_list = args[::2]
        biases_list = args[1::2]

        modeltry = self.build_network(weights_list, biases_list)


        y_pred = modeltry(self.x)



        log_pdf = -0.5 * (self.y - y_pred) ** 2

        posterior_model = tf.math.reduce_mean(log_pdf, axis=(-1, -2))


        return posterior_model


    def plot_neg_log_likelihood(self, neg_log_probs):
        plt.plot(neg_log_probs, '.')
        plt.xlabel("steps")
        plt.ylabel("negative log likelihood")
        plt.show()


    def run_hmc(self, initial_config, **kargs):

        #strategy = tf.distribute.MirroredStrategy()
        #tf.print(tf.config.list_physical_devices('CPU'))
        #tf.print(tf.config.list_physical_devices('GPU'))

        #print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

        #with strategy.scope():

        kernel = tfp.mcmc.HamiltonianMonteCarlo(self.unnormalized_log_prob_normal,
                                             step_size=kargs['step_size'],
                                             num_leapfrog_steps=kargs['num_leapfrog_steps']
                                            )

        '''
        kernel = tfp.mcmc.NoUTurnSampler(self.unnormalized_log_prob_normal,
                                         step_size=kargs['step_size'],
                                         max_tree_depth=kargs['max_tree_depth']
                                        )

        kernel1 = tfp.mcmc.SimpleStepSizeAdaptation(kernel,
                                                   num_adaptation_steps=int(kargs['num_burnin_steps']),
                                                   target_accept_prob=0.9,
                                                   adaptation_rate=2/(kargs['num_burnin_steps']),
                                                  )

        kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(inner_kernel=kernel,
                                                          num_adaptation_steps=int(0.8 * num_burnin_steps)
                                                         )
        '''

        tf.print()
        start = time.perf_counter()

        chain, trace, final_kernel_results = graph_hmc(kernel=kernel,
                                                       current_state=initial_config,
                                                       num_results=kargs['num_results'],
                                                       trace_fn=partial(self.trace_fn, summary_freq=20),
                                                       return_final_kernel_results=True,
                                                       parallel_iterations=kargs['parallel_iterations'],
                                                       num_steps_between_results = 0
                                                      )

        end = time.perf_counter()

        timeused = end - start

        print(f"timeused = {timeused} seconds")


        return chain, trace, final_kernel_results

    def run_hmc_bi_nou(self, initial_config, **kargs):

        #strategy = tf.distribute.MirroredStrategy()
        #tf.print(tf.config.list_physical_devices('CPU'))
        #tf.print(tf.config.list_physical_devices('GPU'))

        #print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

        #with strategy.scope():

        kernel = tfp.mcmc.HamiltonianMonteCarlo(self.unnormalized_log_prob_normal,
                                             step_size=kargs['step_size'],
                                             num_leapfrog_steps=kargs['num_leapfrog_steps']
                                            )
        '''

        kernel = tfp.mcmc.NoUTurnSampler(self.unnormalized_log_prob_normal,
                                         step_size=kargs['step_size'],
                                         max_tree_depth=kargs['max_tree_depth']
                                        )

        '''
        kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(kernel,
                                                          num_adaptation_steps=int(kargs['num_results']),
                                                          target_accept_prob=0.9,
                                                          exploration_shrinkage = 0.1
                                                         )
        '''
        kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(inner_kernel=kernel,
                                                          num_adaptation_steps=int(0.8 * num_burnin_steps)
                                                         )
        '''

        tf.print()
        start = time.perf_counter()

        chain, trace, final_kernel_results = graph_hmc(kernel=kernel,
                                                       current_state=initial_config,
                                                       num_results=kargs['num_results'],
                                                       trace_fn=self.trace_fn_bi_nou,
                                                       return_final_kernel_results=True,
                                                       parallel_iterations=kargs['parallel_iterations'],
                                                       num_steps_between_results = 0
                                                      )

        end = time.perf_counter()

        timeused = end - start

        print(f"timeused = {timeused} seconds")


        return chain, trace, final_kernel_results

    def run_hmc_one_step(self, initial_config, **kargs):

        #strategy = tf.distribute.MirroredStrategy()
        #tf.print(tf.config.list_physical_devices('CPU'))
        #tf.print(tf.config.list_physical_devices('GPU'))

        #print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

        #with strategy.scope():



        kernel = tfp.mcmc.HamiltonianMonteCarlo(self.unnormalized_log_prob_normal,
                                             step_size=kargs['step_size'],
                                             num_leapfrog_steps=kargs['num_leapfrog_steps']
                                            )

        '''
        kernel = tfp.mcmc.NoUTurnSampler(self.unnormalized_log_prob_normal,
                                         step_size=kargs['step_size'],
                                         max_tree_depth=kargs['max_tree_depth']
                                        )

        '''
        kernel = tfp.mcmc.SimpleStepSizeAdaptation(kernel,
                                                   num_adaptation_steps=int(kargs['num_burnin_steps']*0.8),
                                                   target_accept_prob=0.97,
                                                   adaptation_rate=0.01,
                                                  )

        '''
        kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(inner_kernel=kernel,
                                                          num_adaptation_steps=int(0.8 * num_burnin_steps)
                                                         )
        '''

        start = time.perf_counter()
        current_state = initial_config
        previous_kernel_results = kernel.bootstrap_results(initial_config)
        for step in range(kargs['num_burnin_steps'] + kargs['num_results']):
            print('..')
            current_state, previous_kernel_results = kernel.one_step(current_state=initial_config,
                                                                     previous_kernel_results = previous_kernel_results
                                                                    )

        end = time.perf_counter()

        timeused = end - start

        print(f"timeused = {timeused} seconds")


        return


@tf.function
def graph_hmc(*args, **kwargs):
    """Compile static graph for tfp.mcmc.sample_chain.
    Since this is bulk of the computation, using @tf.function here
    signifcantly improves performance (empirically about ~5x).
    """
    return tfp.mcmc.sample_chain(*args, **kwargs)





