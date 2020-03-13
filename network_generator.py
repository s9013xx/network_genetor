import math
import numpy as np
import pandas as pd
import os
import argparse
# from operator import itemgetter
# import cnn
import state_enumerator as se
from state_string_utils import StateStringUtils

class QValues:
    ''' Stores Q_values with helper functions.'''
    def __init__(self):
        self.q = {}

    def load_q_values(self, q_csv_path):
        self.q = {}
        q_csv = pd.read_csv(q_csv_path)
        for row in zip(*[q_csv[col].values.tolist() for col in ['start_layer_type',
                                              'start_layer_depth',
                                              'start_filter_depth',
                                              'start_filter_size',
                                              'start_stride',
                                              'start_image_size',
                                              'start_fc_size',
                                              'start_terminate',
                                              'end_layer_type',
                                              'end_layer_depth',
                                              'end_filter_depth',
                                              'end_filter_size',
                                              'end_stride',
                                              'end_image_size',
                                              'end_fc_size',
                                              'end_terminate',
                                              'utility']]):
            start_state = se.State(layer_type = row[0],
                                   layer_depth = row[1],
                                   filter_depth = row[2],
                                   filter_size = row[3],
                                   stride = row[4],
                                   image_size = row[5],
                                   fc_size = row[6],
                                   terminate = row[7]).as_tuple()
            end_state = se.State(layer_type = row[8],
                                 layer_depth = row[9],
                                 filter_depth = row[10],
                                 filter_size = row[11],
                                 stride = row[12],
                                 image_size = row[13],
                                 fc_size = row[14],
                                 terminate = row[15]).as_tuple()
            utility = row[16]

            if start_state not in self.q:
                self.q[start_state] = {'actions': [end_state], 'utilities': [utility]}
            else:
                self.q[start_state]['actions'].append(end_state)
                self.q[start_state]['utilities'].append(utility)


    def save_to_csv(self, q_csv_path):
        start_layer_type = []
        start_layer_depth = []
        start_filter_depth = []
        start_filter_size = []
        start_stride = []
        start_image_size = []
        start_fc_size = []
        start_terminate = []
        end_layer_type = []
        end_layer_depth = []
        end_filter_depth = []
        end_filter_size = []
        end_stride = []
        end_image_size = []
        end_fc_size = []
        end_terminate = []
        utility = []
        for start_state_list in self.q.keys():
            start_state = se.State(state_list=start_state_list)
            for to_state_ix in range(len(self.q[start_state_list]['actions'])):
                to_state = se.State(state_list=self.q[start_state_list]['actions'][to_state_ix])
                utility.append(self.q[start_state_list]['utilities'][to_state_ix])
                start_layer_type.append(start_state.layer_type)
                start_layer_depth.append(start_state.layer_depth)
                start_filter_depth.append(start_state.filter_depth)
                start_filter_size.append(start_state.filter_size)
                start_stride.append(start_state.stride)
                start_image_size.append(start_state.image_size)
                start_fc_size.append(start_state.fc_size)
                start_terminate.append(start_state.terminate)
                end_layer_type.append(to_state.layer_type)
                end_layer_depth.append(to_state.layer_depth)
                end_filter_depth.append(to_state.filter_depth)
                end_filter_size.append(to_state.filter_size)
                end_stride.append(to_state.stride)
                end_image_size.append(to_state.image_size)
                end_fc_size.append(to_state.fc_size)
                end_terminate.append(to_state.terminate)

        q_csv = pd.DataFrame({'start_layer_type' : start_layer_type,
                              'start_layer_depth' : start_layer_depth,
                              'start_filter_depth' : start_filter_depth,
                              'start_filter_size' : start_filter_size,
                              'start_stride' : start_stride,
                              'start_image_size' : start_image_size,
                              'start_fc_size' : start_fc_size,
                              'start_terminate' : start_terminate,
                              'end_layer_type' : end_layer_type,
                              'end_layer_depth' : end_layer_depth,
                              'end_filter_depth' : end_filter_depth,
                              'end_filter_size' : end_filter_size,
                              'end_stride' : end_stride,
                              'end_image_size' : end_image_size,
                              'end_fc_size' : end_fc_size,
                              'end_terminate' : end_terminate,
                              'utility' : utility})
        q_csv.to_csv(q_csv_path, index=False)

class Network_Generator:
    ''' All Q-Learning updates and policy generator

        Args
            state: The starting state for the QLearning Agent
            q_values: A dictionary of q_values -- 
                            keys: State tuples (State.as_tuple())
                            values: [state list, qvalue list]
            replay_dictionary: A pandas dataframe with columns: 'net' for net strings, and 'accuracy_best_val' for best accuracy
                                        and 'accuracy_last_val' for last accuracy achieved

            output_number : number of output neurons
    '''
    def __init__(self, state_space_parameters, network_number, qstore=None):

        self.state_list = []

        self.state_space_parameters = state_space_parameters

        # Class that will expand states for us
        self.enum = se.StateEnumerator(state_space_parameters)
        self.stringutils = StateStringUtils(state_space_parameters)

        # Starting State
        self.state = se.State('start', 0, 1, 0, 0, state_space_parameters.image_size, 0, 0)# if not state else state
        self.bucketed_state = self.enum.bucket_state(self.state)

        # Cached Q-Values -- used for q learning update and transition
        self.qstore = QValues() if not qstore else qstore
        # self.replay_dictionary = replay_dictionary

        # self.epsilon=epsilon # epsilon: parameter for epsilon greedy strategy
        self.network_number = network_number

    def update_replay_database(self, new_replay_dic):
        self.replay_dictionary = new_replay_dic

    def generate_net(self):
        # Have Q-Learning agent sample current policy to generate a network and convert network to string format
        self._reset_for_new_walk()
        state_list = self._run_agent()
        # state_list = self.stringutils.add_drop_out_states(state_list)
        net_string = self.stringutils.state_list_to_string(state_list)

        print net_string
        # Check if we have already trained this model
        # if net_string in self.replay_dictionary['net'].values:
        #     acc_best_val = self.replay_dictionary[self.replay_dictionary['net']==net_string]['accuracy_best_val'].values[0]
        #     iter_best_val = self.replay_dictionary[self.replay_dictionary['net']==net_string]['iter_best_val'].values[0]
        #     acc_last_val = self.replay_dictionary[self.replay_dictionary['net']==net_string]['accuracy_last_val'].values[0]
        #     iter_last_val = self.replay_dictionary[self.replay_dictionary['net']==net_string]['iter_last_val'].values[0]
        #     acc_best_test = self.replay_dictionary[self.replay_dictionary['net']==net_string]['accuracy_best_test'].values[0]
        #     acc_last_test = self.replay_dictionary[self.replay_dictionary['net']==net_string]['accuracy_last_test'].values[0]
        #     machine_run_on = self.replay_dictionary[self.replay_dictionary['net']==net_string]['machine_run_on'].values[0]
        # else:
        #     acc_best_val = -1.0
        #     iter_best_val = -1.0
        #     acc_last_val = -1.0
        #     iter_last_val = -1.0
        #     acc_best_test = -1.0
        #     acc_last_test = -1.0
        #     machine_run_on = -1.0

        # return (net_string, acc_best_val, iter_best_val, acc_last_val, iter_last_val, acc_best_test, acc_last_test, machine_run_on)

    def save_q(self, q_path):
        self.qstore.save_to_csv(os.path.join(q_path,'q_values.csv'))

    def _reset_for_new_walk(self):
        '''Reset the state for a new random walk'''
        # Architecture String
        self.state_list = []

        # Starting State
        self.state = se.State('start', 0, 1, 0, 0, self.state_space_parameters.image_size, 0, 0)
        self.bucketed_state = self.enum.bucket_state(self.state)

    def _run_agent(self):
        ''' Have Q-Learning agent sample current policy to generate a network
        '''
        while self.state.terminate == 0:
            self._transition_q_learning()

        return self.state_list

    def _transition_q_learning(self):
        ''' Updates self.state according to an epsilon-greedy strategy'''
        if self.bucketed_state.as_tuple() not in self.qstore.q:
            self.enum.enumerate_state(self.bucketed_state, self.qstore.q)

        action_values = self.qstore.q[self.bucketed_state.as_tuple()]

        action = se.State(state_list=action_values['actions'][np.random.randint(len(action_values['actions']))])
        # epsilon greedy choice
        # if np.random.random() < self.epsilon:
        #     action = se.State(state_list=action_values['actions'][np.random.randint(len(action_values['actions']))])
        # else:
        #     max_q_value = max(action_values['utilities'])
        #     max_q_indexes = [i for i in range(len(action_values['actions'])) if action_values['utilities'][i]==max_q_value]
        #     max_actions = [action_values['actions'][i] for i in max_q_indexes]
        #     action = se.State(state_list=max_actions[np.random.randint(len(max_actions))])

        self.state = self.enum.state_action_transition(self.state, action)
        self.bucketed_state = self.enum.bucket_state(self.state)

        self._post_transition_updates()

    def _post_transition_updates(self):
        #State to go in state list
        bucketed_state = self.bucketed_state.copy()

        self.state_list.append(bucketed_state)

    def sample_replay_for_update(self):
        # Experience replay to update Q-Values
        for i in range(self.state_space_parameters.replay_number):
            net = np.random.choice(self.replay_dictionary['net'])
            accuracy_best_val = self.replay_dictionary[self.replay_dictionary['net'] == net]['accuracy_best_val'].values[0]
            accuracy_last_val = self.replay_dictionary[self.replay_dictionary['net'] == net]['accuracy_last_val'].values[0]
            state_list = self.stringutils.convert_model_string_to_states(cnn.parse('net', net))

            state_list = self.stringutils.remove_drop_out_states(state_list)

            # Convert States so they are bucketed
            state_list = [self.enum.bucket_state(state) for state in state_list]

            self.update_q_value_sequence(state_list, self.accuracy_to_reward(accuracy_best_val))

    def accuracy_to_reward(self, acc):
        '''How to define reward from accuracy'''
        return acc

    def update_q_value_sequence(self, states, termination_reward):
        '''Update all Q-Values for a sequence.'''
        self._update_q_value(states[-2], states[-1], termination_reward)
        for i in reversed(range(len(states) - 2)):
            self._update_q_value(states[i], states[i+1], 0)

    def _update_q_value(self, start_state, to_state, reward):
        ''' Update a single Q-Value for start_state given the state we transitioned to and the reward. '''
        if start_state.as_tuple() not in self.qstore.q:
            self.enum.enumerate_state(start_state, self.qstore.q)
        if to_state.as_tuple() not in self.qstore.q:
            self.enum.enumerate_state(to_state, self.qstore.q)

        actions = self.qstore.q[start_state.as_tuple()]['actions']
        values = self.qstore.q[start_state.as_tuple()]['utilities']

        max_over_next_states = max(self.qstore.q[to_state.as_tuple()]['utilities']) if to_state.terminate != 1 else 0

        action_between_states = self.enum.transition_to_action(start_state, to_state).as_tuple()

        # Q_Learning update rule
        values[actions.index(action_between_states)] = values[actions.index(action_between_states)] + \
                                                self.state_space_parameters.learning_rate * (reward + self.state_space_parameters.discount_factor * max_over_next_states - values[actions.index(action_between_states)])

        self.qstore.q[start_state.as_tuple()] = {'actions': actions, 'utilities': values}


def main():

  parser = argparse.ArgumentParser()
  # model_pkgpath = os.path.join(os.path.dirname(__file__),'models')
  # model_choices = next(os.walk(model_pkgpath))[1]

  # parser.add_argument('model',
  #                       help='model package name package should have a model.py,' + 
  #                            'file, hyper_parameters.py file, and a log folder',
  #                       choices=model_choices)

  # parser.add_argument('list_path')
  # parser.add_argument('-eps', '--epsilon', help='For Epsilon Greedy Strategy', type=float)
  parser.add_argument('-nn', '--network_number', type=int, default=10, help='How many models need to generate')

  args = parser.parse_args()

  _model = __import__('models.' + 'cifar10', #args.model,
                        globals(),
                        locals(),
                        ['state_space_parameters', 'hyper_parameters'], 
                        -1)

  network_generator = Network_Generator(_model.state_space_parameters, args.network_number)
  for i in range(args.network_number):
    network_generator.generate_net()
        # qstore=qstore,
        # replay_dictionary=self.replay_dictionary)

# this only runs if the module was *not* imported
if __name__ == '__main__':
    main()  




