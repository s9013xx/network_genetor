import math
# import numpy as np
import pandas as pd
import os
import argparse
from state_string_utils import StateStringUtils
import argparse
import subprocess

def main():
  parser = argparse.ArgumentParser('Network Executor Data Parser')
  parser.add_argument('--cpu', action="store_true", default=False, help='Benchmark using CPU')
  args = parser.parse_args()

  _model = __import__('models.' + 'cifar10', #args.model,
                        globals(),
                        locals(),
                        ['state_space_parameters', 'hyper_parameters'], 
                        -1)
  input_network_csv_path = './network.csv'
  df_net = pd.read_csv(input_network_csv_path)

  created_file = 0
  for net_string in df_net['network']:
    print 'net_string : ', net_string
    if args.cpu:
      command = 'python operation_inference.py --cpu --net_string="%s" > temp' % net_string
    else:
      command = 'python operation_inference.py --net_string="%s" > temp' % net_string
    print "command:", command
    process = subprocess.Popen(command, stdout = subprocess.PIPE, shell = True)
    process.wait()

    time_max = None
    time_min = None
    time_median = None
    time_mean = None
    time_trim_mean = None

    line_count = 0
    tmp_file = open("temp", "r")
    if os.stat("temp").st_size == 0:
      continue
    # line = None
    time_data_ele = None
    for line in tmp_file:
        # print line
        time_data_ele = eval(line)
    time_data_ele.update( {'net_string' : net_string} )
     
    df_ele = pd.DataFrame(data = time_data_ele, index=[0])
    print("time_mean: {} ms".format(time_data_ele['time_mean']))

    if created_file==0: 
        df_ele.to_csv("result.csv", index=False)
        # write_file(df_ele, self.output_exe_path, self.output_exe_file)
        created_file = 1
    else:
        df_ele.to_csv("result.csv", index=False, mode='a', header=False)
        # append_file(df_ele, self.output_exe_path, self.output_exe_file)



    # latency = StateStringUtils(_model.state_space_parameters).convert_model_string_to_states(net_list)
    

  # parser = argparse.ArgumentParser()
  # parser.add_argument('-nn', '--network_number', type=int, default=10, help='How many models need to generate')
  # args = parser.parse_args()

  # _model = __import__('models.' + 'cifar10', #args.model,
  #                       globals(),
  #                       locals(),
  #                       ['state_space_parameters', 'hyper_parameters'], 
  #                       -1)

  # net_list = []
  # for i in range(args.network_number):
  #   network_generator = Network_Generator(_model.state_space_parameters, args.network_number)
  #   net_list.append(network_generator.generate_net())

  # df_net = pd.DataFrame(net_list, columns=['network'])
  # df_net.to_csv('network.csv',index=False)
  # print df_net
  # df_net.to_csv
# this only runs if the module was *not* imported
if __name__ == '__main__':
    main()  




