#!/usr/bin/env python3.6
from PIL import Image
import os
import os.path
import errno
import torch
import codecs
import numpy as np
import sys
import time
import random
import numpy.ma as ma
import copy
import scipy.misc
import scipy.io as scio
from onnx_tf.backend import prepare
import onnx
from lib.network import PoseNet

manualSeed = 46
random.seed(manualSeed)
torch.manual_seed(manualSeed)
np.random.seed(manualSeed)

if torch.cuda.is_available():
  torch.cuda.manual_seed(manualSeed)
  torch.cuda.manual_seed_all(manualSeed)

print('CUDA devices: ', torch.cuda.device_count())
torch.backends.cudnn.benchmark = False

torch.backends.cudnn.deterministic = True
num_objects = 1
num_points = 1000
repeat_epoch = 1

def select_device(device='', apex=False, batch_size=None):
    # device = 'cpu' or '0' or '0,1,2,3'
    cpu_request = device.lower() == 'cpu'
    if device and not cpu_request:  # if device requested other than 'cpu'
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), 'CUDA unavailable, invalid device %s requested' % device  # check availablity

    cuda = False if cpu_request else torch.cuda.is_available()
    if cuda:
        c = 1024 ** 2  # bytes to MB
        ng = torch.cuda.device_count()
        if ng > 1 and batch_size:  # check that batch_size is compatible with device_count
            assert batch_size % ng == 0, 'batch-size %g not multiple of GPU count %g' % (batch_size, ng)
        x = [torch.cuda.get_device_properties(i) for i in range(ng)]
        s = 'Using CUDA ' + ('Apex ' if apex else '')  # apex for mixed precision https://github.com/NVIDIA/apex
        for i in range(0, ng):
            if i == 1:
                s = ' ' * len(s)
            print("%sdevice%g _CudaDeviceProperties(name='%s', total_memory=%dMB)" %
                  (s, i, x[i].name, x[i].total_memory / c))
    else:
        print('Using CPU')

    print('')  # skip a line
    return torch.device('cuda:0' if cuda else 'cpu')

device = select_device(device='cpu')
estimator = PoseNet(num_points = num_points, num_obj = num_objects)
estimator.to(device).eval()                                        
estimator_ONNX_FILE_PATH = './dense-estimator.onnx'

img = torch.zeros((1, 3, 47, 94)).to(device)
points = torch.zeros((1, 1000, 3)).to(device)
choose = torch.zeros((1, 1, 1000), dtype=torch.int64).to(device)
idx = torch.zeros((1, 1), dtype=torch.long).to(device)

torch.onnx.export(estimator, (img, points, choose, idx), estimator_ONNX_FILE_PATH, export_params=True, opset_version=11,input_names=['test_input1','test_input2','test_input3','test_input4' ])

onnx_model = onnx.load(estimator_ONNX_FILE_PATH)
onnx.checker.check_model(onnx_model)
print("Model was successfully converted to ONNX format.")
print("It was saved to", estimator_ONNX_FILE_PATH)
tf_rep = prepare(onnx_model, strict = False)
path = './trained_models/pose.pb'

file = open(path, "wb")
file.write(tf_rep.graph.as_graph_def().SerializeToString())
file.close()

      