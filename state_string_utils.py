import math
import numpy as np
from operator import itemgetter
import tensorflow as tf
import state_enumerator as se
import random
import time
import numpy
from scipy import stats
import json

class StateStringUtils:
    ''' Contains all functions dealing with converting nets to net strings
        and net strings to state lists.
    '''
    def __init__(self, state_space_parameters):
        self.image_size = state_space_parameters.image_size
        self.output_number = state_space_parameters.output_states
        self.enum = se.StateEnumerator(state_space_parameters)

    def add_drop_out_states(self, state_list):
        ''' Add drop out every 2 layers and after each fully connected layer
        Sets dropout rate to be between 0 and 0.5 at a linear rate
        '''
        new_state_list = []
        number_fc = len([state for state in state_list if state.layer_type == 'fc'])
        number_gap = len([state for state in state_list if state.layer_type == 'gap'])
        number_drop_layers = (len(state_list) - number_gap - number_fc)/2 + number_fc
        drop_number = 1
        for i in range(len(state_list)):
            new_state_list.append(state_list[i])
            if ((((i+1) % 2 == 0 and i != 0) or state_list[i].layer_type == 'fc')
                and state_list[i].terminate != 1
                and state_list[i].layer_type != 'gap'
                and drop_number <= number_drop_layers):
                drop_state = state_list[i].copy()
                drop_state.filter_depth = drop_number
                drop_state.fc_size = number_drop_layers
                drop_state.layer_type = 'dropout'
                drop_number += 1
                new_state_list.append(drop_state)

        return new_state_list

    def remove_drop_out_states(self, state_list):
        new_state_list = []
        for state in state_list:
            if state.layer_type != 'dropout':
                new_state_list.append(state)
        return new_state_list


    def state_list_to_string(self, state_list):
        '''Convert the list of strings to a string we can train from according to the grammar'''
        out_string = ''
        strings = []
        i = 0
        while i < len(state_list):
            state = state_list[i]
            if self.state_to_string(state):
                strings.append(self.state_to_string(state))
            i += 1
        return str('[' + ', '.join(strings) + ']')

    def state_to_string(self, state):
        ''' Returns the string asociated with state.
        '''
        if state.terminate == 1:
            return 'SM(%i)' % (self.output_number)
        elif state.layer_type == 'conv':
            return 'C(%i,%i,%i,%i,%i,%i)' % (state.filter_depth, state.filter_size, state.stride, state.conv_padding, state.conv_act, state.conv_bias)
        elif state.layer_type == 'gap':
            return 'GAP(%i)' % (self.output_number)
        elif state.layer_type == 'pool':
            return 'P(%i,%i,%i)' % (state.filter_size, state.stride, state.pool_padding)
        elif state.layer_type == 'fc':
            return 'FC(%i,%i,%i)' % (state.fc_size, state.fc_act, state.fc_bias)
        elif state.layer_type == 'dropout':
            return 'D(%i,%i)' % (state.filter_depth, state.fc_size) ##SUPER BAD i am using fc_size and filter depth -- should fix later
        return None

    def convert_model_string_to_states(self, parsed_list, start_state=None):
        '''Takes a parsed model string and returns a recursive list of states.'''

        states = [start_state] if start_state else [se.State('start', 0, 1, 0, 0, self.image_size, 0, 0)]
        activation_list = ['None', 'tf.nn.relu']
        first_layer = 1

        batchsize = random.randrange(1, 129, 1)
        input_image_size = random.randrange(1, 513, 1)
        input_image_channels = random.randrange(1, 3, 1)
        
        total_conv_filters = 0
        total_conv_kernelsizes = 0
        total_conv_strides = 0
        total_conv_paddings = 0
        total_conv_acts = 0
        total_conv_bias = 0

        total_pool_sizes = 0
        total_pool_strides = 0
        total_pool_paddings = 0

        total_fc_units = 0
        total_fc_acts = 0
        total_fc_bias = 0

        time_list = []
        time_max = None
        time_min = None
        time_median = None
        time_mean = None
        time_trim_mean = None

        tf.reset_default_graph()
        op = None
        for layer in parsed_list:
            if layer[0] == 'conv':
                if first_layer == 1:
                    first_layer = 0
                    op = tf.Variable(tf.random_normal([batchsize, input_image_size, input_image_size, input_image_channels]))

                op = tf.layers.conv2d(op, filters=layer[1], kernel_size=[layer[2], layer[2]], strides=(layer[3], layer[3]), 
                    padding=('SAME' if layer[4] ==1 else 'VALID'), activation=eval(activation_list[layer[5]]), 
                    use_bias=layer[6], name='convolution_%d'%(states[-1].layer_depth + 1))

                total_conv_filters = total_conv_filters + layer[1]
                total_conv_kernelsizes = total_conv_kernelsizes + layer[2]**2
                total_conv_strides = total_conv_strides + layer[3]**2
                total_conv_paddings = total_conv_paddings + layer[4]
                total_conv_acts = total_conv_acts + layer[5]
                total_conv_bias = total_conv_bias + layer[6]

                states.append(se.State(layer_type='conv',
                                    layer_depth=states[-1].layer_depth + 1,
                                    filter_depth=layer[1],
                                    filter_size=layer[2],
                                    stride=layer[3],
                                    image_size=states[-1].image_size,
                                    fc_size=0,
                                    terminate=0,
                                    conv_padding=layer[4],
                                    conv_act=layer[5],
                                    conv_bias=layer[6],
                                    pool_padding=0,
                                    fc_act=0,
                                    fc_bias=0,
                                    state_list=0))
            # elif layer[0] == 'gap':
            #     states.append(se.State(layer_type='gap',
            #                             layer_depth=states[-1].layer_depth + 1,
            #                             filter_depth=0,
            #                             filter_size=0,
            #                             stride=0,
            #                             image_size=1,
            #                             fc_size=0,
            #                             terminate=0))
            elif layer[0] == 'pool':
                if first_layer == 1:
                    first_layer = 0
                    input_image_size = states[-1].image_size ** 2
                    op = tf.Variable(tf.random_normal([batchsize, input_image_size, input_image_size, input_image_channels]))

                op = tf.layers.max_pooling2d(op, pool_size=(layer[1], layer[1]), strides=(layer[2], layer[2]), 
                    padding=('SAME' if layer[3]==1 else 'VALID'), name = 'pooling_%d'%(states[-1].layer_depth + 1))
                
                total_pool_sizes = total_pool_sizes + layer[1]**2
                total_pool_strides = total_pool_strides + layer[2]**2
                total_pool_paddings = total_pool_paddings + layer[3]

                states.append(se.State(layer_type='pool',
                                    layer_depth=states[-1].layer_depth + 1,
                                    filter_depth=0,
                                    filter_size=layer[1],
                                    stride=layer[2],
                                    image_size=self.enum._calc_new_image_size(states[-1].image_size, layer[1], layer[2]),
                                    fc_size=0,
                                    terminate=0,
                                    conv_padding=0,
                                    conv_act=0,
                                    conv_bias=0,
                                    pool_padding=layer[3],
                                    fc_act=0,
                                    fc_bias=0,
                                    state_list=0))
            elif layer[0] == 'fc':
                if first_layer == 1:
                    first_layer = 0
                    input_image_size = states[-1].image_size ** 2
                    op = tf.Variable(tf.random_normal([batchsize, input_image_size*input_image_size*input_image_channels]))

                op = tf.layers.dense(inputs=op, units=layer[1], kernel_initializer=tf.ones_initializer(), 
                    activation=eval(activation_list[layer[2]]), use_bias=layer[3], name = 'dense_%d'%(states[-1].layer_depth + 1))
                
                total_fc_units = total_fc_units + layer[1]
                total_fc_acts = total_fc_acts + layer[2]
                total_fc_bias = total_fc_bias + layer[3]

                states.append(se.State(layer_type='fc',
                                    layer_depth=states[-1].layer_depth + 1,
                                    filter_depth=len([state for state in states if state.layer_type == 'fc']),
                                    filter_size=0,
                                    stride=0,
                                    image_size=0,
                                    fc_size=layer[1],
                                    terminate=0,
                                    conv_padding=0,
                                    conv_act=0,
                                    conv_bias=0,
                                    pool_padding=0,
                                    fc_act=layer[2],
                                    fc_bias=layer[3],
                                    state_list=0))
            # elif layer[0] == 'dropout':
            #     states.append(se.State(layer_type='dropout',
            #                             layer_depth=states[-1].layer_depth,
            #                             filter_depth=layer[1],
            #                             filter_size=0,
            #                             stride=0,
            #                             image_size=states[-1].image_size,
            #                             fc_size=layer[2],
            #                             terminate=0))
            # elif layer[0] == 'softmax':
            #     termination_state = states[-1].copy() if states[-1].layer_type != 'dropout' else states[-2].copy()
            #     termination_state.terminate=1
            #     termination_state.layer_depth += 1
            #     states.append(termination_state)
        sess = tf.Session()
        if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
            init = tf.initialize_all_variables()
        else:
            init = tf.global_variables_initializer()
        sess.run(init)
        # Warm-up run
        for _ in range(5):#args.iter_warmup):
            sess.run(op)
        # Benchmark run
        for _ in range(10):#args.iter_benchmark):
            start_time = time.time()
            sess.run(op)
            time_list.append(((time.time()-start_time) * 1000))

        np_array_parameters = np.array(time_list)
        time_max = numpy.amax(np_array_parameters)
        time_min = numpy.amin(np_array_parameters)
        time_median = numpy.median(np_array_parameters)
        time_mean = numpy.mean(np_array_parameters)
        time_trim_mean = stats.trim_mean(np_array_parameters, 0.1)

        
        result_dict = {
            'batchsize' : batchsize,
            'input_image_size' : input_image_size**2,
            'input_image_channels' : input_image_channels,

            'total_conv_filters' : total_conv_filters,
            'total_conv_kernelsizes' : total_conv_kernelsizes,
            'total_conv_strides' : total_conv_strides,
            'total_conv_paddings' : total_conv_paddings,
            'total_conv_acts' : total_conv_acts,
            'total_conv_bias' : total_conv_bias,

            'total_pool_sizes' : total_pool_sizes,
            'total_pool_strides' : total_pool_strides,
            'total_pool_paddings' : total_pool_paddings,

            'total_fc_units' : total_fc_units,
            'total_fc_acts' : total_fc_acts,
            'total_fc_bias' : total_fc_bias,

            'time_max' : time_max,
            'time_min' : time_min,
            'time_median' : time_median,
            'time_mean' : time_mean,
            'time_trim_mean' : time_trim_mean,
        }
        print result_dict

        # return states



