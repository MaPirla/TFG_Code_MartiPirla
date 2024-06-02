from collections import defaultdict
import numpy as np

class QlearningAgent():
    """
    ## Q-Learning Agent
    Generic Q-learning agent, to use it, you have to call the take_action_learn function, that will return the action the
    model wants to do. Then, the game has to return a reward into the update_qtable fuction, in order to learn from the
    action made before.
    When the model has learned, you can use take_action in order to use always the best action.
    
    ### Atributes:
    - q_table: Table that uses the algorithm in order to learn.
    - accumulated_reward: The reward that the model has made until now.
    - current_state_action: pair state-action that represents in the last state and the action the model took.
    - exploration_factor: ratio of actions that the model will take randomly
    - learning_rate: Weight of the reward of the current action over the previous rewards.
    
    """
    def __init__(self, exploration_factor: float=0.2, exploration_factor_progression: float=1, discount_rate: float =0.9, learning_rate: float =0.4, print_debug:bool = False, biases:dict = {}) -> None:

        self.print_debug = print_debug
        self.q_table = defaultdict(float)
        self.accumulated_reward = 0
        self.biases = biases
        self.discount_rate = discount_rate
        self.current_state_action = None
        self.exploration_factor = exploration_factor
        self.exploration_factor_progression = exploration_factor_progression
        self.learning_rate = learning_rate

    def take_action_learn(self, state, available_actions):
        """
            Make the model choose an action on the current state.
            This function will apply the exploration rate, returning sometimes a random action.
        """
        if self.print_debug: print("current_state setted to", self.current_state_action)
        selected_action = ""
        if np.random.uniform(0.,1.) < self.exploration_factor:
            if self.print_debug: print("exploratory")
            selected_action = np.random.choice(available_actions)
        else:
            if self.print_debug: print("Player looks for best value")
            best_action = ""
            best_value = -float("inf")
            for action in available_actions:
                if self.q_table_element((state, action)) > best_value:
                    best_value = self.q_table_element((state, action))
                    best_action = action
            selected_action = best_action
        self.current_state_action = (state, selected_action)
        return selected_action

    def take_action(self, state, available_actions):
        """
            Make the model choose an action on the current state.
            This function will not apply the exploration rate, returning allways the action considered as best.
        """
        if self.print_debug: print("current_state setted to", self.current_state_action)
        selected_action = ""
        if self.print_debug: print("Player looks for best value")
        best_action = ""
        best_value = -float("inf")
        for action in available_actions:
            if self.q_table_element((state, action)) > best_value:
                best_value = self.q_table_element((state, action))
                best_action = action
        selected_action = best_action
        #self.current_state_action = (state, selected_action)
        return selected_action

    def update_qtable(self, reward: float, new_state, action_list):
        """
            After taking an action, updates the value of that action in that state with the reward.
        """
        if self.current_state_action != None:
            if self.print_debug: print("Inmediate reward:", reward)

            old_value = self.q_table_element(self.current_state_action)#self.q_table[self.current_state_action]
            action_value_list = []
            for action in action_list:
                action_value_list.append(self.q_table_element((new_state, action)))#self.q_table[(new_state, action)])
            max_q = max(action_value_list)
            if self.print_debug: print("update", old_value, "= ", old_value, "+", self.learning_rate, "*(", reward, "+", self.discount_rate, "*", max_q, "-", old_value, ")")
            self.q_table[self.current_state_action] = old_value + self.learning_rate*(reward + self.discount_rate*max_q - old_value)
            if self.print_debug: print("The new value of", self.current_state_action, "is ", old_value)
            self.accumulated_reward += reward
    
    def clear_next_episode(self):
        """
            Resets the last state to start the next game, also reduces the exploration_factor if the exploration_factor_progression is < 1.
        """
        self.current_state_action = None
        self.exploration_factor *= self.exploration_factor_progression
    
    def clear_table(self):
        """
            Clears the Qtable.
        """
        self.q_table = defaultdict(float)

    def q_table_element(self, state_action):
        if state_action not in self.q_table.keys():
            if state_action[1] in self.biases.keys():
                self.q_table[state_action] = self.biases[state_action[1]]
        return self.q_table[state_action]