# SIR on networks from multivariate distributions

In this project we apply the SIR (Susceptible Infected Recovered) model, on networks that were created by random sampling on multivariate distributions such as Lognormal or Weibull. The multivariate distributions are utilized to define the degrees of the 2 different groups of the network's population. To create the networks, we use Gibbs Sampling, a Markov Chain Monte Carlo algorithm, which produces a sequence that approximates the joint distribution. Then we continue by providing the sequence to create the networks, applying two models, the original configuration model, and configuration model relying on preferential attachment.

### Configuration model

Configuration model, is a model that utilizes a sequence of the expected degrees of a network to create links. The model assumes that every pair of nodes that has available links (given the expected degrees) has the same probability to be formed.

### Configuration model with preferential attachment

This model is similar to the previous, with the difference that the probability of the connection of a link between a pair of nodes i, j is proportional to the expected degree of the node j. 

Finally we apply stochastic SIR simulations to the produced networks.