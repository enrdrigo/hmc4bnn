Tensorflow implememtation of a Bayesian Neural Network.
The training data are first fitted with Adam or SGD algorithm and then the optimized configuration is set to be the initial state of a Hamiltonian Markov Chain for the estimation of the uncertainaty of the model given the dataset.

TODO: improve parallelization
TODO: solve and analyze the step_size issue
