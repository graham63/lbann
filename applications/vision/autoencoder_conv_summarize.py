import argparse
import lbann
import lbann.models
import lbann.contrib.args
import lbann.contrib.models.wide_resnet
import lbann.contrib.launcher
import data.cifar10
import data.imagenet

# Command-line arguments
desc = ('Construct and run ResNet on ImageNet-1K data. '
        'Running the experiment is only supported on LC systems.')
parser = argparse.ArgumentParser(description=desc)
lbann.contrib.args.add_scheduler_arguments(parser)
parser.add_argument(
    '--job-name', action='store', default='lbann_image_ae', type=str,
    help='scheduler job name (default: lbann_resnet)')
parser.add_argument(
    '--width', action='store', default=1, type=float,
    help='Wide ResNet width factor (default: 1)')
parser.add_argument(
    '--bn-statistics-group-size', action='store', default=1, type=int,
    help=('Group size for aggregating batch normalization statistics '
          '(default: 1)'))
parser.add_argument(
    '--warmup', action='store_true', help='use a linear warmup')
parser.add_argument(
    '--mini-batch-size', action='store', default=256, type=int,
    help='mini-batch size (default: 256)', metavar='NUM')
parser.add_argument(
    '--num-epochs', action='store', default=90, type=int,
    help='number of epochs (default: 90)', metavar='NUM')
parser.add_argument(
    '--num-classes', action='store', default=1000, type=int,
    help='number of ImageNet classes (default: 1000)', metavar='NUM')
parser.add_argument(
    '--random-seed', action='store', default=0, type=int,
    help='random seed for LBANN RNGs', metavar='NUM')
parser.add_argument(
    '--dataset', action='store', default='imagenet', type=str,
    help='dataset to use; \"cifar10\" or \"imagenet\"')
parser.add_argument(
    '--data-reader-percent', action='store',
    default=1.0, type=float,
    help='the percent of the data to use (default: 1.0)', metavar='NUM')
lbann.contrib.args.add_optimizer_arguments(parser, default_learning_rate=0.1)
args = parser.parse_args()

# Due to a data reader limitation, the actual model realization must be
# hardcoded to 1000 labels for ImageNet; 10 for CIFAR10.
dataset = args.dataset;
if dataset == 'imagenet':
    num_labels=1000
elif dataset == 'cifar10':
    num_labels=10
else:
    print("Dataset must be cifar10 or imagenet. Try again.")
    exit()

# Construct layer graph
input_ = lbann.Input(name='input')
image = lbann.Identity(input_, name='images')
dummy = lbann.Dummy(input_, name='labels')

# Encoder

conv1 = lbann.Convolution(image,
                          name="conv1",
                          num_dims=2,
                          num_output_channels=16,
                          conv_dims=[3,3],
                          conv_pads=[0,0],
                          conv_strides=[1,1],
                          has_bias=True,
                          has_vectors=True)

relu1 = lbann.Relu(conv1, name="relu1")

pool1 = lbann.Pooling(relu1,
                      name="pool1",
                      num_dims=2,
                      pool_dims=[2,2],
                      pool_pads=[0,0],
                      pool_strides=[1,1],
                      pool_mode="max",
                      has_vectors=True)


conv2 = lbann.Convolution(pool1,
                          name="conv2",
                          num_dims=2,
                          num_output_channels=8,
                          conv_dims=[3,3],
                          conv_pads=[0,0],
                          conv_strides=[1,1],
                          has_bias=True,
                          has_vectors=True)

relu2 = lbann.Relu(conv2, name="relu2")

pool2 = lbann.Pooling(relu2,
                      name="pool2",
                      num_dims=2,
                      pool_dims=[2,2],
                      pool_pads=[0,0],
                      pool_strides=[1,1],
                      pool_mode="max",
                      has_vectors=True)

conv3 = lbann.Convolution(pool2,
                          name="conv3",
                          num_dims=2,
                          num_output_channels=8,
                          conv_dims=[3,3],
                          conv_pads=[0,0],
                          conv_strides=[1,1],
                          has_bias=True,
                          has_vectors=True)

relu3 = lbann.Relu(conv3, name="relu3")

pool3 = lbann.Pooling(relu3,
                      name="pool3",
                      num_dims=2,
                      pool_dims=[2,2],
                      pool_pads=[0,0],
                      pool_strides=[1,1],
                      pool_mode="max",
                      has_vectors=True)

unpool3 = lbann.Unpooling(pool3,
                          name="unpool3",
                          num_dims=2,
                          pooling_layer=pool3)

deconv3 = lbann.Deconvolution(unpool3,
                              name="deconv3",
                              num_dims=2,
                              num_output_channels=8,
                              conv_dims=[3,3],
                              conv_pads=[0,0],
                              conv_strides=[1,1],
                              has_bias=True,
                              has_vectors=True)

relu4 = lbann.Relu(deconv3, name="relu4")

unpool2 = lbann.Unpooling(relu4,
                          name="unpool2",
                          num_dims=2,
                          pooling_layer=pool2)

deconv2 = lbann.Deconvolution(unpool2,
                              name="deconv2",
                              num_dims=2,
                              num_output_channels=16,
                              conv_dims=[3,3],
                              conv_pads=[0,0],
                              conv_strides=[1,1],
                              has_bias=True,
                              has_vectors=True)

relu5 = lbann.Relu(deconv2, name="relu5")

unpool1 = lbann.Unpooling(relu5,
                          name="unpool1",
                          num_dims=2,
                          pooling_layer=pool1)

deconv1 = lbann.Deconvolution(unpool1,
                              name="deconv1",
                              num_dims=2,
                              num_output_channels=3,
                              conv_dims=[3,3],
                              conv_pads=[0,0],
                              conv_strides=[1,1],
                              has_bias=True,
                              has_vectors=True)

relu6 = lbann.Relu(deconv1, name="relu6")

decode1 = lbann.FullyConnected(relu6,
                               name="decode1",
                               hint_layer=image,
                               num_neurons=784,
                               has_bias=True)

reconstruction = lbann.Sigmoid(decode1,
                               name="reconstruction")


# Reconstruction
mean_squared_error = lbann.MeanSquaredError([reconstruction, image],
                             name="mean_squared_error")

layer_term = lbann.LayerTerm(mean_squared_error)
obj = lbann.ObjectiveFunction(layer_term)

metrics = [lbann.Metric(mean_squared_error, name="mean squared error")]

img_strategy = lbann.AutoencoderStrategy(
    input_layer_name='input',
    num_tracked_images=20)

summarize_images = lbann.CallbackSummarizeImages(
    selection_strategy=img_strategy,
    image_source_layer_name='conv1',
    epoch_interval=2)

summarize_input_layer = lbann.CallbackSummarizeImages(
    selection_strategy=img_strategy,
    image_source_layer_name=image.name,
    epoch_interval=10000)

callbacks = [lbann.CallbackPrint(),
             lbann.CallbackTimer(),
             summarize_images,
             summarize_input_layer]

layers = list(lbann.traverse_layer_graph(input_))
model = lbann.Model(args.mini_batch_size,
                    args.num_epochs,
                    layers=layers,
                    objective_function=obj,
                    metrics=metrics,
                    callbacks=callbacks,
                    random_seed=args.random_seed,
                    summary_dir="/g/g13/graham63/workspace/code/lbann/event_files")

# Setup optimizer
opt = lbann.contrib.args.create_optimizer(args)

# Setup data reader
num_classes=min(args.num_classes, num_labels)

if dataset == "cifar10":
    data_reader = data.cifar10.make_data_reader(num_classes=num_classes)
else:
    data_reader = data.imagenet.make_data_reader(num_classes=num_classes)

# Setup trainer
trainer = lbann.Trainer(disable_cuda=True)

# Run experiment
kwargs = lbann.contrib.args.get_scheduler_kwargs(args)
kwargs['lbann_args'] = '--data_reader_percent='+str(args.data_reader_percent)

lbann.contrib.launcher.run(trainer, model, data_reader, opt,
                           job_name=args.job_name,
                           **kwargs)