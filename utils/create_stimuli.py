
import numpy as np
import math
def create_stimuli(hor, nEpisode):
    # Difficulty Level (how distinguishable are the stimuli)
    difficulty = [0.01, 0.05, 0.1, 0.15, 0.2]
    couples = np.array([0,1])

    nrep = int(nEpisode / len(difficulty))

    nrep2 = (hor + 1) * nrep
    nTrials2 = int(nrep2 * len(difficulty))
    trialListValues = np.empty((nTrials2,2**(hor+1))); trialListValues[:] = np.nan

    #gain/loss for each trial
    gl1 = 0.3
    if hor>1:
        gl1 = 0.19

    for n, diff in zip(np.arange(0,len(difficulty))+1, difficulty):
        # mean value is limited
        #mu_lim = (diff / 2) + gl1*h
        
        # equalizing stimuli as a function of difficulty but varying the mean
        #mu0 = (np.random.randint(1,round((1-2*mu_lim)*100),(nrep,1)) + math.floor(mu_lim*100)) / 100
        mu0 = round(np.random.uniform(gl1*hor+0.2/2, 1-gl1*hor-0.2/2), 3)
        sl1=(-1)**np.random.randint(0,2,(nrep,1)) # randomize small-large stimuli 
        sl2=(-1)**np.random.randint(0,2,(nrep,1)) # randomize small-large stimuli
        sl3=(-1)**np.random.randint(0,2,(nrep,1)) # randomize small-large stimuli 
        
        # first trial
        t=1
        trialListValues[np.arange((hor+1)*(n-1)*nrep+t,(hor+1)*n*nrep+1,(hor+1))-1, couples[0]:couples[1]+1] = mu0+sl1*diff*[-1/2, 1/2] 
        if hor>0:
            t=2
            trialListValues[np.arange((hor+1)*(n-1)*nrep+t,(hor+1)*n*nrep+1,(hor+1))-1,couples[0]:couples[1]+1] = mu0+sl1*gl1+sl2*diff*[-1/2,1/2]#LEFT
            trialListValues[np.arange((hor+1)*(n-1)*nrep+t,(hor+1)*n*nrep+1,(hor+1))-1,2:4]=mu0-sl1*gl1+sl2*diff*[-1/2, 1/2]#RIGHT
        if hor>1:
            t=3
            trialListValues[np.arange((hor+1)*(n-1)*nrep+t,(hor+1)*n*nrep+1,(hor+1))-1,0:2] = mu0+sl1*gl1+sl2*gl1+sl3*diff*[-1/2,1/2]#LEFT LEFT
            trialListValues[np.arange((hor+1)*(n-1)*nrep+t,(hor+1)*n*nrep+1,(hor+1))-1,2:4]=mu0+sl1*gl1-sl2*gl1+sl3*diff*[-1/2, 1/2]#LEFT RIGHT
            trialListValues[np.arange((hor+1)*(n-1)*nrep+t,(hor+1)*n*nrep+1,(hor+1))-1,4:6] = mu0-sl1*gl1+sl2*gl1+sl3*diff*[-1/2,1/2]#RIGHT LEFT
            trialListValues[np.arange((hor+1)*(n-1)*nrep+t,(hor+1)*n*nrep+1,(hor+1))-1,6:8]=mu0-sl1*gl1-sl2*gl1+sl3*diff*[-1/2, 1/2]#RIGHT RIGHT
    return trialListValues

def get_best_reward(states):
    col_index = 0
    total_reward = 0
    for i in range(len(states)):
        state = states[i]
        
        r = False
        if state[col_index] > state[col_index+1]:
            if i+1 == len(states):
                total_reward += state[col_index]
            else:
                total_reward += state[col_index+1]
                r=True
        else:
            if i+1 == len(states):
                total_reward += state[col_index+1]
                r=True
            else:
                total_reward += state[col_index]

        col_index *= 2
        if r: col_index += 2
        
    return total_reward


def get_worst_reward(states):
    col_index = 0
    total_reward = 0
    for i in range(len(states)):
        state = states[i]
        
        r = False
        if state[col_index] > state[col_index+1]:
            if i+1 == len(states):
                total_reward += state[col_index+1]
                r=True
            else:
                total_reward += state[col_index]
                
        else:
            if i+1 == len(states):
                total_reward += state[col_index]
                
            else:
                total_reward += state[col_index+1]
                r=True

        col_index *= 2
        if r: col_index += 2
        
    return total_reward


def create_episode_performance(states, reward):
    min_rew = get_worst_reward(states)
    max_rew = get_best_reward(states)
    return (reward-min_rew)/(max_rew-min_rew)