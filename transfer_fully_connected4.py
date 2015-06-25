import numpy
import theano
from simplelearn.utils import safe_izip
from copy_dataset import Dataset
from simplelearn.data.mnist import load_mnist
from simplelearn.formats import DenseFormat
from copy_nodes import RescaleImage, FormatNode, Conv2dLayer, SoftmaxLayer, CrossEntropy, Misclassification, AffineLayer
from copy_training7 import Sgd, SemiSgd, SemiSgdPlus, SavesAtMinimum, AverageMonitor, ValidationCallback, SavesAtMinimum, StopsOnStagnation, LimitsNumEpochs
from simplelearn.io import SerializableModel
#from extensions import PaulSgdParameterUpdater
#from copy_training import SemiSgd
import matplotlib.pyplot as plt
import time
import pdb


###### HERE IS THE MAIN EXAMPLE ##########

training_set, testing_set = load_mnist()

training_tensors = [t[:50000, ...] for t in training_set.tensors]  # the first 50000 examples
validation_tensors = [t[50000:, ...] for t in training_set.tensors]  # the remaining 10000 examples
training_set, validation_set = [Dataset(tensors=t,
                                        names=training_set.names,
                                        formats=training_set.formats)
                                for t in (training_tensors, validation_tensors)]

training_iter = training_set.iterator(iterator_type='sequential', batch_size=100)

image_node, label_node = training_iter.make_input_nodes()

float_image_node = RescaleImage(image_node)

input_shape = float_image_node.output_format.shape
conv_input_node = FormatNode(input_node=float_image_node,  # axis order: batch, rows, cols
                             output_format=DenseFormat(axes=('b', 'c', '0', '1'),  # batch, channels, rows, cols
                                                       shape=(input_shape[0],  # batch size (-1)
                                                              1,               # num. channels
                                                              input_shape[1],  # num. rows (28)
                                                              input_shape[2]), # num cols (28)
                                                       dtype=None),  # don't change the input's dtype
                             axis_map={'b': ('b', 'c')})  # split batch axis into batch & channel axes

layers = [conv_input_node]

for _ in range(2):  # repeat twice
    layers.append(AffineLayer(input_node=layers[-1],  # last element of <layers>
                              output_format=DenseFormat(axes=('b', 'f'),  # axis order: (batch, feature)
                                                        shape=(-1, 10),   # output shape: (variable batch size, 10 classes)
                                                        dtype=None)    # don't change the input data type
                              ))

layers.append(SoftmaxLayer(input_node=layers[-1],
                           output_format=DenseFormat(axes=('b', 'f'),  # axis order: (batch, feature)
                                                     shape=(-1, 10),   # output shape: (variable batch size, 10 classes)
                                                     dtype=None),      # don't change the input data type
                           ))  # collapse the channel, row, and column axes to a single feature axis

rng = numpy.random.RandomState(34523)  # mash the keypad with your forehead to come up with a suitable seed
softmax_layer = layers[-1]
affine_weights_symbol = softmax_layer.affine_node.linear_node.params
affine_weights_values = affine_weights_symbol.get_value()
std_deviation = .05
affine_weights_values[...] = rng.standard_normal(affine_weights_values.shape) * std_deviation
affine_weights_symbol.set_value(affine_weights_values)

for i in range(1,3):
    rng = numpy.random.RandomState(34523)  # mash the keypad with your forehead to come up with a suitable seed
    affine_layer = layers[i]
    affine_weights_symbol = affine_layer.affine_node.linear_node.params
    affine_weights_values = affine_weights_symbol.get_value()
    std_deviation = .05
    affine_weights_values[...] = rng.standard_normal(affine_weights_values.shape) * std_deviation
    affine_weights_symbol.set_value(affine_weights_values)

loss_node = CrossEntropy(softmax_layer, label_node)

param_symbols = []

# add the filters and biases from each convolutional layer
for i in range(1,4):
    param_symbols.append(layers[i].affine_node.linear_node.params)
    param_symbols.append(layers[i].affine_node.bias_node.params)

scalar_loss_symbol = loss_node.output_symbol.mean()  # the mean over the batch axis. Very important not to use sum().
gradient_symbols = [theano.gradient.grad(scalar_loss_symbol, p) for p in param_symbols]  # derivatives of loss w.r.t. each of the params

# packages chain of nodes from the uint8 image_node up to the softmax_layer, to be saved to a file.
model = SerializableModel([image_node], [softmax_layer])

# A Node that outputs 1 if output_node's label diagrees with label_node's label, 0 otherwise.
misclassification_node = Misclassification(softmax_layer, label_node)

#
# Callbacks to feed the misclassification rate (MCR) to after each epoch:
#

# Prints misclassificiation rate (must be a module-level function to be pickleable).
def print_misclassification_rate(values, _):  # ignores 2nd argument (formats)
    print("Misclassification rate: %s" % str(values))

# Saves <model> to file "best_model.pkl" if MCR is the best yet seen.
saves_best = SavesAtMinimum(model, "./best_model.pkl")

# Raises a StopTraining exception if MCR doesn't decrease for more than 10 epochs.
training_stopper = StopsOnStagnation(max_epochs=10, min_proportional_decrease=0.0)

# Measures the average misclassification rate over some dataset
misclassification_rate_monitor = AverageMonitor(misclassification_node.output_symbol,
                                                misclassification_node.output_format,
                                                callbacks=[print_misclassification_rate,
                                                           saves_best,
                                                           training_stopper])

validation_iter = validation_set.iterator(iterator_type='sequential', batch_size=100)

# Gets called by trainer between training epochs.
validation_callback = ValidationCallback(inputs=[image_node.output_symbol, label_node.output_symbol],
                                         input_iterator=validation_iter,
                                         monitors=[misclassification_rate_monitor])

trainer = SemiSgdPlus([image_node, label_node],
              training_iter,
              param_symbols,
              monitors=[],
              training_set = training_set,
              gradient=gradient_symbols,
              loss_function = scalar_loss_symbol,
              learning_rate=.02,
              momentum=0.8,
              epoch_callbacks=[validation_callback,  # measure validation misclassification rate, quit if it stops falling
                               LimitsNumEpochs(100)])  # perform no more than 100 epochs

start_time = time.time()
_classification_errors = trainer.train()
print _classification_errors
elapsed_time = time.time() - start_time

plt.plot(_classification_errors)
plt.title('Learning Curve for S2GD+ (Learning rate = 0.02, Momentum = 0.8)')
plt.xlabel('Epochs')
plt.ylabel('Classification error')
plt.show()

print "The time elapsed for training is ", elapsed_time

