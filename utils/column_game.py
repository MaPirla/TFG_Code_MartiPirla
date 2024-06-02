from utils.create_stimuli import create_stimuli, create_episode_performance, get_best_reward, get_worst_reward
from utils.qlearningAgent import QlearningAgent
import numpy as np
import pandas as pd


class ColumnGame():
    def __init__(self, agent: QlearningAgent, horizon: int, nEpisodes: int, print_debug: bool = False, action_set = ["B", "S"]):
        self.agent = agent
        self.stimuli = create_stimuli(horizon, nEpisodes)
        self.current_episode = 0
        self.horizon = horizon
        self.acumulated_reward = 0
        self.print_debug = print_debug
        self.action_set = action_set
            #'R', # Choose the column on the right
            #'L', # Choose the column on the left
            #'B', # Choose the bigger column
            #'S', # Choose the smaller column
            #'1', # Choose the first apearing column
            #'2'  # Choose the second column to apear
            

    def start_game(self):
        col_pos = 0
        states = self.stimuli[self.current_episode*(self.horizon+1):(self.current_episode+1)*(self.horizon+1)]

        col_sizes = states[0, col_pos:col_pos+2]
        state = 2**(self.horizon+1)
        big_first = np.random.choice([True, False])
        episode_reward = 0
        entries = []
        
        df= pd.DataFrame(columns=["horizon", "nte", "chosen_big", "chosen_right", "chosen_first", "performance"])
        
        for i in range(self.horizon+1):

            if self.print_debug: print("\n\nState: ", col_sizes, ", ", state)
            
            #Makes the agent choose an action
            action = self.agent.take_action_learn(state, self.action_set)
            
            if self.print_debug: print("the agent chose the column that is ", action)
            
            #Transform the different actions into a value representing if the chosen column is the big or small one
            col = 0
            match action:
                case "B":
                    col = 0
                case "S":
                    col = 1
                case "R":
                    #if state[1] != "R": col = 1
                    if state[0] > state[1]:
                        col = 1
                case "L":
                    if state[0] < state[1]:
                        col = 1
                case "1":
                    if not big_fist:
                        col = 1
                case "2":
                    if big_fist:
                        col = 1
            
            #Based on which column is chosen, calculates the next state
            state_transition = 2**(self.horizon-i)
            if col == 0:
                state += state_transition
            else:
                state -= state_transition

            #Determines the reward that the action gives
            reward = 0
            chosen_col = 0
            if ((col == 1) and (col_sizes[0] > col_sizes[1])) or ((col == 0) and (col_sizes[0] < col_sizes[1])):
                reward = col_sizes[1]
                chosen_col = 1
            else:
                reward = col_sizes[0]

            episode_reward += reward
            if self.print_debug and col == 0: print("The agent chose the bigger column, , giving a reward of", col_sizes[chosen_col])
            if self.print_debug and col == 1: print("The agent chose the smaller column, giving a reward of", col_sizes[chosen_col])

            #Get the pair of columns for the next state
            col_pos *= 2
            if (col == 1) == (col_sizes[0] > col_sizes[1]): #isRight
                col_pos +=2
            if self.horizon > i:
                col_sizes = states[i+1, col_pos:col_pos+2]
            
            
            self.acumulated_reward += reward
            if self.print_debug: print("The current acumulated reward is", self.acumulated_reward)
            
            #Makes the agent update the q_table with the reward
            self.agent.update_qtable(reward, state, self.action_set)

            #Calculates the proportion of reward that is currently achieved
            performance = create_episode_performance(states, episode_reward)

            #We create an entry as a df and append it into a list
            new_record = pd.DataFrame([{"horizon": self.horizon, "nte": i+1, "chosen_big": col == 0, "chosen_right": (col == 1) == (col_sizes[0] > col_sizes[1]), "chosen_first": True, "performance": performance}])
            entries.append(new_record)
        
        if self.print_debug: print("END OF THE EPISODE _______________________________________________\n\n")
        self.agent.clear_next_episode()
        #Makes the next call to this function run the next episode
        self.current_episode += 1
        #form the dataframe of all the entries
        return pd.concat(entries, ignore_index=True)
