# https://pastebin.com/aJG3Ukmn

#!/usr/bin/python
# Trivial toy implementation of forward-backward
# Follows Jason Eisner's SpreadSheet-based teaching tool,
# "An Interactive Spreadsheet for Teaching the Forward-Backward Algorithm (2002)"
#  http://www.cs.jhu.edu/~jason/papers/
#  and produces the same results.
import pprint

class forwardBackward( ):
    def __init__(self, obs, init_mat, state2obs, state2state, final_mat):
        """Initialize the probabilitiy matrices."""
        self.alphas      = []
        self.betas       = []
        self.init_mat    = init_mat
        self.state2obs   = state2obs
        self.state2state = state2state
        self.final_mat   = final_mat
        self.obs         = obs
        #Temporary receptacles for re-estimated values
        self.re_init_mat    = {}
        self.re_state2obs   = {}
        self.re_state2state = {}
        self.re_final_mat   = {}
    
    def _init_alphas( self ):
        """
           Initialize the forward alpha probabilities.
        """
        for state in self.init_mat:
            prob   = self.init_mat[state] * self.state2obs[state][self.obs[0]]
            if len(self.alphas)>0:
                self.alphas[0][state] = [prob, "START %s" % (state)]
            else:
                self.alphas = [ {} for i in range(len(self.obs)) ]
                self.alphas[0] = { state : [prob, "START %s" % (state)] }
        
        return
    
    def _init_betas( self ):
        """
           Initialize the beta backward probabilities.
        """
        self.betas = [ {} for i in range(len(self.obs)) ]
        for state in self.final_mat:
            self.betas[len(self.obs)-1][state] = [self.final_mat[state], "END %s %s" % (state, state)]
        return
    
    def _calc_forward_alphas( self ):
        """
           Calculate all of the forward alpha probabilities
           The partial best paths:
           GIVEN the observation for stage N
           FOREACH state Y AT stage N,
             FOREACH state X AT stage N-1
               CALCULATE prob(state Y | observation) *
                         prob(state Y | previous state at stage N-1 was X) *
                         prob(partial best path from stage N-1)
             RECORD total probabiity of path to state Y AT stage N
        """
        for i,o in enumerate(self.obs):
            if i==0: continue
            max_prob  = 0; max_state = 0
            for curr in self.state2obs:
                curr_prob = 0; max_prob = 0; max_state = 0
                xmax = 0; tmps = None
                for prev in self.state2state:
                    val, states = self.alphas[i-1][prev]
                    subtot = self.state2obs[curr][self.obs[i]] \
                        * self.state2state[prev][curr] \
                        * val
                    curr_prob += subtot
                    if subtot > xmax:
                        xmax = subtot
                        tmps = states
                if curr_prob==0.0:
                    print ("ERROR: curr_prob==0.0.  Failed to reach final state for the current iteration.")
                    sys.exit()
                if curr_prob > max_prob:
                    max_state = tmps
                    max_prob  = curr_prob
                self.alphas[i][curr] = [max_prob, "%s %s" % (max_state, curr)]
        return
    
    def _calc_backward_betas( self ):
        """
           Same process as _calc_forward_alphas, but in reverse.
        """
        for i in range(len(self.obs)-2,-1,-1):
            max_prob = 0; max_state = 0
            for curr in self.state2obs:
                curr_prob = 0; max_prob = 0; max_state = 0
                xmax = 0; tmps = None
                for prev in self.state2state:
                    val, states = self.betas[i+1][prev]
                    subtot = self.state2obs[prev][self.obs[i+1]] \
                        * self.state2state[curr][prev] \
                        * val
                    curr_prob += subtot
                    if subtot > xmax:
                        xmax = subtot
                        tmps = states
                if curr_prob > max_prob:
                    max_state = tmps
                    max_prob  = curr_prob
                self.betas[i][curr] = [max_prob, "%s %s" % (max_state, curr)]
        return
    
    def best_path( self ):
        """Viterbi best path through the lattice."""
        top_prob = 0.0
        best_path = None
        for state in self.alphas[len(self.alphas)-1]:
            val, states = self.alphas[len(self.alphas)-1][state]
            if val > top_prob:
                top_prob = val
                best_path = states
        print ("%s\t%s" %(best_path,str(top_prob)))
        return top_prob
    
    def _reestimate_probs( self ):
        """
           Reestimate all the transition probabilities.
        
           This is a 3-step process,
        
           1. Compute the alpha-beta values, normalize and store them
           2. Re-normalize the init, s2s, s2o and final matrices using the
               results of 1.
        """
        
        reest = {}
        gtot  = 0.0
        ssum  = 0.0
        state_totals = {}
        
        #Iterate through the alphas and betas and compute 
        # the combined alpha-beta values
        for j,val in enumerate(self.alphas):
            sum = 0.0
            ab_vals = {}
            for key in self.alphas[j]:
                alpha = self.alphas[j][key]
                beta  = self.betas[j][key]
                alphaBeta = alpha[0] * beta[0]
                ab_vals[key]  = alphaBeta
                sum      += alphaBeta
            ssum = sum
            for key in ab_vals:
                total = ab_vals[key] / sum
                if key in reest: 
                    if self.obs[j] in reest[key]:
                        reest[key][self.obs[j]] += total
                    else:
                        reest[key][self.obs[j]]  = total
                else:
                    reest[key] = { self.obs[j]:total }
                    
                if key in state_totals: 
                    state_totals[key] += total
                else:
                    state_totals[key]  = total
                gtot += total
        
        #Compute the reestimated state-2-observation matrix
        for key in reest:
            for s1 in reest[key]:
                reestimated = reest[key][s1] / state_totals[key]
                if key in self.re_state2obs: 
                    self.re_state2obs[key][s1] = reestimated
                else:
                    self.re_state2obs[key] = {s1:reestimated}
        
        #Compute the reestimated init matrix
        for key in self.alphas[0]:
            alpha     = self.alphas[0][key]
            beta      = self.betas[0][key]
            alphaBeta = alpha[0] * beta[0]
            tri       = alphaBeta / ssum
            self.re_init_mat[key] = tri
        
        #Compute the reestimated state-2-state matrix
        transitions = {}
        for j,o in [(j,o) for j,o in enumerate(self.obs) if j > 0]:   # skip first element
            for s1 in self.state2state:
                for s2 in  self.state2state[s1]:
                    al    = self.alphas[j-1][s1][0]
                    be    = self.betas[j][s2][0]
                    trans = self.state2state[s1][s2]
                    conf  = self.state2obs[s2][self.obs[j]]
                    stot  = al * be * trans * conf / ssum
                    if s1 in transitions: 
                        if s2 in transitions[s1]: 
                            transitions[s1][s2] += stot
                        else:
                            transitions[s1][s2] = stot
                    else:
                        transitions[s1] = {s2:stot}
        
        for s1 in transitions:
            for s2 in transitions[s1]:
                newP = transitions[s1][s2] / state_totals[s1]
                if s1 in self.re_state2state: 
                    self.re_state2state[s1][s2] = newP
                else:
                    self.re_state2state[s1] = {s2:newP}
        
        #Compute the reestimated final matrix
        for key in self.alphas[len(self.alphas)-1]:
            alpha     = self.alphas[len(self.alphas)-1][key]
            beta      = self.betas[len(self.alphas)-1][key]
            alphaBeta = alpha[0] * beta[0]
            tri = alphaBeta / ssum / state_totals[key]
            self.re_final_mat[key] = tri
        
        self.init_mat    = self.re_init_mat
        self.state2obs   = self.re_state2obs
        self.state2state = self.re_state2state
        self.final_mat   = self.re_final_mat
        #Reset the alphas and betas
        self.alphas      = []
        self.betas       = []
        return
    
    def iterate( self, n_iter=5, ratio=4e-20, verbose=False ):
        """Run the algorithm iteratively until convergence."""
        prev_likelihood = 9999999
        likelihoodHistory = []
        for i in range(n_iter):
            if verbose:
                print ("Iteration: {}\nState2Obs".format(i))
                pprint.pprint(self.state2obs)
                print ("State2State")
                pprint.pprint(self.state2state)
            self._init_alphas()
            self._init_betas()
            self._calc_forward_alphas()
            self._calc_backward_betas()
            
            likelihood = self.best_path()
            if abs(likelihood - prev_likelihood) < ratio:
                print ("Achieved convergence ratio:", ratio,"; Stopping at iteration: %d." % i)
                break
            elif verbose:
                print ("Likelihood change:", abs(likelihood - prev_likelihood))
            self._reestimate_probs()  
            likelihoodHistory.append(likelihood)
            prev_likelihood = likelihood
        return {'init_mat':self.init_mat, 'state2obs':self.state2obs, 'state2state':self.state2state, 
                'final_mat':self.final_mat, 'likelihoodHistory':likelihoodHistory}

def setupHMM():
    obs = '2,3,3,2,3,2,3,2,2,3,1,3,3,1,1,1,2,1,1,1,3,1,2,1,1,1,2,3,3,2,3,2,2'.split(',')
    init_mat    = {'C':.5, 'H':.5}
    state2obs   = {'C':{'1':.7,'2':.2,'3':.1},'H':{'1':.1,'2':.7,'3':.2}}
    state2state = {'C':{'C':.8,'H':.1},'H':{'C':.1,'H':.8}}
    final_mat   = {'C':.1,'H':.1}
    return obs, init_mat, state2obs, state2state, final_mat

if __name__=="__main__":
    obs, init_mat, state2obs, state2state, final_mat = setupHMM()
    fb = forwardBackward(obs, init_mat, state2obs, state2state, final_mat)
    HMM_params = fb.iterate(n_iter=10, ratio=1e-15, verbose=True)
    print()
    pprint.pprint(HMM_params)

import matplotlib.pyplot as plt
plt.plot(HMM_params['likelihoodHistory'])
plt.xlabel('iteration')
plt.ylabel('likelihood')
plt.show()
