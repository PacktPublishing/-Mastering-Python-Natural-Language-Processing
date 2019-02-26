#https://en.wikipedia.org/wiki/Viterbi_algorithm

obs = ('normal', 'cough', 'headache', 'normal', 'normal')
states = ('Healthy', 'Sick')
start_p = {'Healthy': 0.3, 'Sick': 0.7}
trans_p = {
   'Healthy' : {'Healthy': 0.9, 'Sick': 0.1},
   'Sick' : {'Healthy': 0.2, 'Sick': 0.8}
   }
obs_p = {
   'Healthy' : {'normal': 0.8, 'cough': 0.15, 'headache': 0.05},
   'Sick' : {'normal': 0.1, 'cough': 0.6, 'headache': 0.3}
   }

import numpy as np

periods = 5
health = ['Healthy']
probability = [1.0]
randomNumbers = np.random.rand(periods)
for i in range(periods):
    if trans_p[health[i]]['Healthy'] < randomNumbers[i]:
        health.append('Healthy')
    else:
        health.append('Sick')
    probability.append((trans_p[health[i]][health[i+1]]))

print('health sequence: {}'.format(health))
print('probabilities: {}'.format(probability))
print('sequence probability = {}'.format(np.prod(probability)))

def getPath(V):
    """Backtrack through lattice to retrieve path to optimal solution"""
    opt = []
    max_prob = max(value["prob"] for value in V[-1].values())
    previous = None
    # Get most probable state and its backtrack
    for st, data in V[-1].items():
        if data["prob"] == max_prob:
            opt.append(st)
            previous = st
            break
    # Follow the backtrack till the first observation
    for t in range(len(V) - 2, -1, -1):
        opt.insert(0, V[t + 1][previous]["prior"])
        previous = V[t + 1][previous]["prior"]
    return opt, max_prob

def viterbi(obs, states, start_p, trans_p, obs_p):
    """Construct the lattice using the forward algorithm"""
    V = [{}]
    # Initialize the lattice
    for st in states:
        V[0][st] = {"prob": start_p[st] * obs_p[st][obs[0]], "prior": None}
    # Run Viterbi when t > 0 to incrementally create the lattice
    for t in range(1, len(obs)):
        V.append({})
        for st in states:
            max_tr_prob = V[t-1][states[0]]["prob"] * trans_p[states[0]][st]
            prev_st_selected = states[0]
            for prev_st in states[1:]:
                tr_prob = V[t-1][prev_st]["prob"] * trans_p[prev_st][st]
                if tr_prob > max_tr_prob:
                    max_tr_prob = tr_prob
                    prev_st_selected = prev_st
                    
            max_prob = max_tr_prob * obs_p[st][obs[t]]
            V[t][st] = {"prob": max_prob, "prior": prev_st_selected}
                    
    opt, max_prob = getPath(V)
    print ('The steps of states are ' + ' '.join(opt) + ' with highest probability of %s' % max_prob)
    return V, opt

V, opt = viterbi(obs, states, start_p, trans_p, obs_p)

def roundN(x, d = 6):
    return round(x,d)
    
import matplotlib.pyplot as plt

def graphDisplay(outfileName = None):	
    if outfileName == None:
        plt.show()
    else:
        plt.savefig(outfileName)

def graphNetworkHMM(title, getValues, outfileName = None):
    nsteps = len(obs)
    plt.close()
    fig, ax = plt.subplots()
    plt.rcParams["figure.figsize"] = [2 * (nsteps + 1),6]
    plt.xlim(-1 - 0.5, nsteps - 0.5)
    plt.ylim(-1, 2)
    plt.title(title)
    plt.axis('off')
    optNumbers = [revDict[o] for o in opt]
    for i,x in enumerate(optNumbers[:-1]):
            ax.arrow(i, x, 0.65, 0.75 * (optNumbers[i+1] - x), head_width=0.05, head_length=0.1, fc='k', ec='k')
    for j, s in enumerate(states):
        plt.text(-1, j, s, verticalalignment='center', horizontalalignment='left')
    for i, o in enumerate(obsLabels):
        plt.text(i, 1.75, o, verticalalignment='center', horizontalalignment='center')
        for j, s in enumerate(states):
            ax.add_artist(plt.Rectangle((i-.25, j-.25), 0.5, 0.5, color='white', fill=True))
            ax.add_artist(plt.Rectangle((i-.25, j-.25), 0.5, 0.5, color='black', fill=False))
            plt.text(i, j, roundN(getValues(i,o,j,s)), verticalalignment='center', horizontalalignment='center')
    graphDisplay(outfileName)
		
def getValues1(i,o,j,s):
    return obs_p[s][obs[i]]

def getValues0(i,o,j,s):
    return V_df[obsLabels[i]][s]

graphNetworkHMM("conditional probability for each state", getValues0, outfileName='graphics/HMMConditionalProb.png')
graphNetworkHMM("cumulative probabilities at each state", getValues1, outfileName='graphics/HMMCumulativeProb.png')    


############################

import numpy as np
import math

def graphNetwork(plt, title, getValues, placeEdges, states, ncols=4, outfileName = None):
    nrows = len(states)
    plt.close()
    plt.rcParams["figure.figsize"] = [2 * (ncols + 1),2 * nrows]
    fig, ax = plt.subplots()
    plt.xlim(-1, ncols)
    plt.ylim(-.5, nrows - 0.5)
    plt.title(title)
    plt.axis('off')
    placeEdges(ax, nrows, ncols)
    for j, s in enumerate(states):
        plt.text(-1, j, s, verticalalignment='center', horizontalalignment='left')
    for i, o in enumerate(range(ncols)):
        #plt.text(i, 1.75, o, verticalalignment='center', horizontalalignment='center')
        for j, s in enumerate(range(nrows)):
            ax.add_artist(plt.Circle((i, j), 0.25, color='white', fill=True))
            ax.add_artist(plt.Circle((i, j), 0.25, color='black', fill=False))
            plt.text(i, j, getValues(i,o,j,s), verticalalignment='center', horizontalalignment='center')
    graphDisplay(outfileName)
	
def placeHMMEdges(ax, nrows, ncols):
    for i,x in enumerate(range(ncols)):
        ax.arrow(i, 1, 0.65, 0, head_width=0.05, head_length=0.1, fc='k', ec='k')
        ax.arrow(i, 1, 0, -.65, head_width=0.05, head_length=0.1, fc='k', ec='k')

def getHMMLabels(i,o,j,s):
    return eval("r'$\ {0}_{1}$'".format(['O','t'][j], i))

import matplotlib.pyplot as plt
graphNetwork(plt, "Hidden Markov Model", getHMMLabels, placeHMMEdges, ['observed','hidden'], outfileName='graphics/HiddenMarkovModel.png')

import numpy as np
import math

def graphSetup():
    fig = plt.figure()
    ax = plt.axes()
    x = np.linspace(-6, 6, 100)
    plt.grid(linestyle='--')
    return x

x = graphSetup()
plt.title('logistic function')
plt.plot(x,  [1 / (1 + math.exp(-xx)) for xx in x])
plt.text(-4.5, .8, r'$y = \frac{1}{1 + exp(-x)}$', fontsize=20)
graphDisplay(outfileName='graphics/LogisticFunction.png')

x = graphSetup()
plt.title('RELU')
plt.plot(x,  [max(0,xx) for xx in x])
plt.text(-4.5, .8, r'$y = max(0, x)$', fontsize=20)
graphDisplay(outfileName='graphics/ReLuFunction.png')


def placeNNEdges(ax, nrows, ncols):
    for j1 in range(nrows):
        ax.arrow(-.65, j1, .25, 0, head_width=0.05, head_length=0.1, fc='k', ec='k')
    for i,x in enumerate(range(ncols - 1)):
        for j1 in range(nrows):
            for j2 in range(nrows):
                ax.arrow(i, j1, .65, (j2-j1) * .75, head_width=0.05, head_length=0.1, fc='k', ec='k')

def getNNLabels(i,o,j,s):
    return eval("r'$\ {0}_{1}$'".format(['I','H1','H2','H3'][i], j))

inputLabels = [r'$\ Input_{0}$'.format(i) for i in range(4) ]
graphNetwork(plt, "Neural Network", getNNLabels, placeNNEdges, inputLabels, 3, outfileName='graphics/NeuralNetwork.png')


