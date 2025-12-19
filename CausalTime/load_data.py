import numpy as np
import sys
sys.path.append('../')

def read_causaltime(type):

    data_name = type
    data = np.load('./' + data_name + '/gen_data.npy')
    GC = np.load('./' + data_name + '/graph.npy')

    Sample_num = data.shape[0]
    Time_step = data.shape[1]
    Node_num = data.shape[2]

    data = data.reshape(Sample_num * Time_step, Node_num)

    return data, GC
