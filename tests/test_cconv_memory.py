
# This model shows a form of question answering with memory. It binds two 
# features (color and shape) by circular convolution and stores them in a memory 
# population. It then provides a cue to the model at a later time to 
# determine the feature bound to that cue by deconvolution. This model exhibits 
# better cognitive ability since the answers to the questions are provided at a 
# later time and not at the same time as the questions themselves.

# **Note: A simplified method of building the model using the spa (semantic 
# pointer architecture) package in Nengo 2.0 is shown in the 
# spa_question-memory.py file in the same folder. 

# This model has parameters as described in the book, with memory population 
# having 1000 neurons over 20 dimensions. The memory population is capable of 
# storing a vector over time and it uses an integrator network to do so as 
# discussed in the book.

#Setup the environment
import numpy as np
import nengo

dim=2         # Number of dimensions 
N_input=300    # Number of neurons in population
N_conv=70      # Number of neurons per dimension in bind/unbind populations
N_mem=50       # Number of neurons per dimension in memory population

pulse_interval = 1.0
#Creating the vocabulary
rng = np.random.RandomState(7)  

model = nengo.Network(seed=12)  
with model:
    #Ensembles
    A = nengo.Ensemble(n_neurons=N_input, dimensions=dim, label='U')
    B = nengo.Ensemble(n_neurons=N_input, dimensions=dim, label='V')
    C = nengo.Ensemble(n_neurons=N_input, dimensions=dim, label='cue')
    D = nengo.Ensemble(n_neurons=N_input, dimensions=dim, label='bound')
    E = nengo.Ensemble(n_neurons=N_input, dimensions=dim, label='output')
    
    #Creating memory population and connecting ensemble D to it
    tau = 0.4
    memory = nengo.networks.EnsembleArray(n_neurons=N_mem, n_ensembles=dim, 
                                            label='Memory')
    nengo.Connection(memory.output, memory.input, synapse=tau)     
    nengo.Connection(D, memory.input, synapse=0.01)
    
    #Creating the Bind network
    bind = nengo.networks.CircularConvolution(n_neurons=N_conv, dimensions=dim)
    nengo.Connection(A, bind.A)
    nengo.Connection(B, bind.B)
    nengo.Connection(bind.output, D) 
    
    #Creating the Unbind network
    unbind = nengo.networks.CircularConvolution(n_neurons=N_conv,  
                                        dimensions=dim, invert_a=True)
    nengo.Connection(C, unbind.A)
    nengo.Connection(memory.output, unbind.B)
    nengo.Connection(unbind.output, E)

    n_items = 3
    zero = np.zeros(dim)

    U = nengo.dists.UniformHypersphere(surface=False).sample(n_items, dim, rng=rng)
    U_data = { i*pulse_interval: U[i, :] for i in range(U.shape[0]) }
    U_data[U.shape[0]*pulse_interval] = zero
    u = nengo.Node(output=nengo.processes.Piecewise(U_data))
    
    V = nengo.dists.UniformHypersphere(surface=False).sample(n_items, dim, rng=rng)
    V_data = { i*pulse_interval: V[i, :] for i in range(V.shape[0]) }
    V_data[V.shape[0]*pulse_interval] = zero
    
    v = nengo.Node(output=nengo.processes.Piecewise(V_data))

    F = nengo.dists.UniformHypersphere(surface=False).sample(1, dim, rng=rng)
    #Defining inputs
    print(U_data)
    cue_input = [zero, zero, zero, zero, U[1,:], zero, F[0], zero, U[2,:]]
    print(cue_input)
    cue = nengo.Node(output=nengo.processes.PresentInput(cue_input, presentation_time=pulse_interval))

    #Connecting input to ensembles
    nengo.Connection(u, A)
    nengo.Connection(v, B)
    nengo.Connection(cue, C)

    
