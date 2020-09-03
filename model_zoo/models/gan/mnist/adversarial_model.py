################################################################################
# Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# Written by the LBANN Research Team (B. Van Essen, et al.) listed in
# the CONTRIBUTORS file. <lbann-dev@llnl.gov>
#
# LLNL-CODE-697807.
# All rights reserved.
#
# This file is part of LBANN: Livermore Big Artificial Neural Network
# Toolkit. For details, see http://software.llnl.gov/LBANN or
# https://github.com/LLNL/LBANN.
#
# Licensed under the Apache License, Version 2.0 (the "Licensee"); you
# may not use this file except in compliance with the License.  You may
# obtain a copy of the License at:
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the license.
#
# FIXME: Add description
# adversarial_model.py -
#
################################################################################

import os.path
import google.protobuf.text_format as txtf
import sys
import argparse
import lbann
import lbann.models
import lbann.contrib.args
import lbann.contrib.models.wide_resnet
import lbann.contrib.launcher

# Get relative path to data readers
current_file = os.path.realpath(__file__)
current_dir = os.path.dirname(current_file)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))), 'data_readers'))

data_reader_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))), 'data_readers')

print("data_reader_path = " + data_reader_dir)

# Command-line arguments
# FIXME: I don't know what this is
desc = ('Construct and run ?. '
        'Running the experiment is only supported on LC systems.')
parser = argparse.ArgumentParser(description=desc)
lbann.contrib.args.add_scheduler_arguments(parser)
parser.add_argument(
    '--job-name', action='store', default='lbann_image_ae', type=str,
    help='scheduler job name (default: lbann_GAN)')
parser.add_argument(
    '--mini-batch-size', action='store', default=32, type=int,
    help='mini-batch size (default: 32)', metavar='NUM')
parser.add_argument(
    '--num-epochs', action='store', default=1, type=int,
    help='number of epochs (default: 1)', metavar='NUM')
parser.add_argument(
    '--num-classes', action='store', default=1000, type=int,
    help='number of ImageNet classes (default: 1000)', metavar='NUM')
parser.add_argument(
    '--random-seed', action='store', default=0, type=int,
    help='random seed for LBANN RNGs', metavar='NUM')
parser.add_argument(
    '--super-steps', action='store', default=100000, type=int,
    help='number of super steps', metavar='NUM')
parser.add_argument(
    '--data-reader', action='store',
    default='data_reader_mnist.prototext', type=str,
    help='scheduler job name (default: data_reader_mnist.prototext)')
lbann.contrib.args.add_optimizer_arguments(parser, default_learning_rate=0.1)
args = parser.parse_args()


# Start of layers

# Construct layer graph
input_ = lbann.Input(name='input')
data = lbann.Split(input_, name='data')
label = lbann.Split(input_, name='label')

# Divide mini-batch samples into two halves

mb_index = lbann.MiniBatchIndex(name="mb_index")

mb_size = lbann.MiniBatchSize(name="mb_size")

mb_factor = lbann.Divide([mb_index, mb_size], name="mb_factor")

in_second_half_scalar = lbann.Round(mb_factor, name="in_second_half_scalar")

in_second_half_scalar3d = lbann.Reshape(in_second_half_scalar,
                                        name="in_second_half_scalar3d",
                                        dims=[1, 1, 1])

in_second_half = lbann.Tessellate(in_second_half_scalar3d,
                                  name="in_second_half",
                                  hint_layer="data")

in_first_half = lbann.LogicalNot(in_second_half, name="in_first_half")

# ZERO
zero_data = lbann.Multiply([data, in_second_half], name="zero_data")

# Generator Path

# NOISE
noise = lbann.Gaussian(name="noise",
                       mean=0.0,
                       stdev=1.0,
                       neuron_dims=100)

gen_fc_weights = lbann.Weights(name="gen_fc_weights",
                               optimizer=lbann.NoOptimizer(),
                               initializer=lbann.GlorotNormalInitializer())

fc1 = lbann.FullyConnected(noise,
                           name="fc1",
                           #weights=gen_fc_weights,
                           num_neurons=256,
                           has_bias=True)

fc1_relu = lbann.LeakyRelu(fc1, name="fc1_relu")

fc1_bn = lbann.BatchNormalization(fc1_relu,
                                  name="fc1_bn",
                                  decay=0.9,
                                  scale_init=1.0,
                                  bias_init=0.0,
                                  epsilon=0.00005)

fc2 = lbann.FullyConnected(fc1_bn,
                           name="fc2",
                           #weights=gen_fc_weights,
                           num_neurons=512,
                           has_bias=True)

fc2_relu = lbann.LeakyRelu(fc2, name="fc2_relu")

fc2_bn = lbann.BatchNormalization(fc2_relu,
                                  name="fc2_bn",
                                  decay=0.9,
                                  scale_init=1.0,
                                  bias_init=0.0,
                                  epsilon=0.00005)

fc3 = lbann.FullyConnected(fc2_bn,
                           name="fc3",
                           #weights=gen_fc_weights,
                           num_neurons=1024,
                           has_bias=True)

fc3_relu = lbann.LeakyRelu(fc3, name="fc3_relu")

fc3_bn = lbann.BatchNormalization(fc3_relu,
                                  name="fc3_bn",
                                  decay=0.9,
                                  scale_init=1.0,
                                  bias_init=0.0,
                                  epsilon=0.00005)

fc4 = lbann.FullyConnected(fc3_bn,
                           name="fc4",
                           #weights=gen_fc_weights,
                           num_neurons=784,
                           has_bias=True)


fc4_tanh = lbann.Tanh(fc4, name="fc4_tanh")

# Reshape for discriminator
reshape1 = lbann.Reshape(fc4_tanh,
                         name="reshape1",
                         num_dims=3,
                         dims=[1, 28, 28])

# Zero
zero_fake = lbann.Multiply([reshape1, in_first_half], name="zero_fake")

# Sum
sum = lbann.Sum([zero_data, zero_fake], name="sum")

# Discriminator Model

dis_flatten_weights = lbann.Weights(name="dis_flatten_weights",
                                    optimizer=lbann.NoOptimizer(),
                                    initializer=lbann.HeNormalInitializer())

dis_flatten_proxy = lbann.FullyConnected(sum,
                                         name="dis_flatten_proxy",
                                         weights=dis_flatten_weights,
                                         num_neurons=784,
                                         has_bias=True)

dis_fc1_weights = lbann.Weights(name="dis_fc1_weights",
                                optimizer=lbann.NoOptimizer(),
                                initializer=lbann.GlorotNormalInitializer())

dis_fc1_proxy = lbann.FullyConnected(dis_flatten_proxy,
                                     name="dis_fc1_proxy",
                                     weights=dis_fc1_weights,
                                     num_neurons=512,
                                     has_bias=True)

dis_fc1_relu = lbann.LeakyRelu(dis_fc1_proxy, name="dis_fc1_relu")

dis_fc2_weights = lbann.Weights(name="dis_fc2_weights",
                                optimizer=lbann.NoOptimizer(),
                                initializer=lbann.GlorotNormalInitializer())

dis_fc2_proxy = lbann.FullyConnected(dis_fc1_relu,
                                     name="dis_fc2_proxy",
                                     weights=dis_fc2_weights,
                                     num_neurons=256,
                                     has_bias=True)

dis_fc2_relu = lbann.LeakyRelu(dis_fc2_proxy, name="dis_fc2_relu")

dis_fc3_weights = lbann.Weights(name="dis_fc3_weights",
                                optimizer=lbann.NoOptimizer(),
                                initializer=lbann.GlorotNormalInitializer())

dis_fc3_proxy = lbann.FullyConnected(dis_fc2_relu,
                                     name="dis_fc3_proxy",
                                     weights=dis_fc3_weights,
                                     num_neurons=2,
                                     has_bias=True)

sigmoid2 = lbann.Sigmoid(dis_fc3_proxy, name="sigmoid2")

# Softmax

prob = lbann.Softmax(sigmoid2, name="prob")

binary_cross_entropy = lbann.BinaryCrossEntropy([prob, label],
                                                name="binary_cross_entropy")

accuracy = lbann.CategoricalAccuracy([prob, label],
                                     name="accuracy")

layer_list = list(lbann.traverse_layer_graph(input_))

# Set up objective function
layer_term = lbann.LayerTerm(binary_cross_entropy)
l2_reg = lbann.L2WeightRegularization(scale=0.0004)
obj = lbann.ObjectiveFunction([layer_term, l2_reg])

# Metrics
metrics = [lbann.Metric(accuracy, name="accuracy", unit="%")]

# Callbacks
callbacks = [lbann.CallbackPrint(),
             lbann.CallbackTimer(),
             lbann.CallbackDumpOutputs(directory=".",
                               layers=[fc4_tanh, sum],
                               execution_modes="test"),
             lbann.CallbackSaveImages(layers="image reconstruction",
                                      image_format="jpg")]

# Setup Model
model = lbann.Model(args.num_epochs,
                    layers=layer_list,
                    objective_function=obj,
                    metrics=metrics,
                    callbacks=callbacks,
                    summary_dir=".")

# Setup optimizer
opt = lbann.contrib.args.create_optimizer(args)

# Setup data reader
data_reader_file = args.data_reader
data_reader_proto = lbann.lbann_pb2.LbannPB()
with open(data_reader_file, 'r') as f:
    txtf.Merge(f.read(), data_reader_proto)
data_reader_proto = data_reader_proto.data_reader


# Setup trainer
trainer = lbann.Trainer(mini_batch_size=args.mini_batch_size)

# Run experiment
kwargs = lbann.contrib.args.get_scheduler_kwargs(args)

lbann.contrib.launcher.run(trainer, model, data_reader_proto, opt,
                           job_name=args.job_name,
                           **kwargs)
