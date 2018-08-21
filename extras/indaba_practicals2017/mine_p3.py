# Import TensorFlow and some other libraries we'll be using.
import datetime
import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

# Import Matplotlib and set some defaults
from matplotlib import pyplot as plt
plt.ioff()
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# Download the MNIST dataset onto the local machine.
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# Conv layer forward pass
def convolutional_forward(X, W, b, filter_size, depth, stride, padding):
    # X has size [batch_size, input_width, input_height, input_depth]
    # W has shape [filter_size, filter_size, input_depth, depth]
    # b has shape [depth]
    batch_size, input_width, input_height, input_depth = X.shape

    # Check that the weights are of the expected shape
    assert W.shape == (filter_size, filter_size, input_depth, depth)

    # QUESTION: Calculate the width and height of the output
    # output_width = ...
    # output_height = ...
    #
    # ANSWER:
    output_width = (input_width - filter_size + 2 * padding) / stride + 1
    output_height = (input_height - filter_size + 2 * padding) / stride + 1
    ####

    # Apply padding to the width and height dimensions of the input
    X_padded = np.pad(X, ((0, 0), (padding, padding), (padding, padding), (0, 0)), 'constant')

    # Allocate the output Tensor
    out = np.zeros((batch_size, output_width, output_height, depth))

    # NOTE: There is a more efficient way of doing a convolution, but this most
    # clearly illustrates the idea.

    for i in range(output_width):  # Loop over the output width dimension
        for j in range(output_height):  # Loop over the output height dimension

            # Select the current block in the input that the filter will be applied to
            block_width_start = i * stride
            block_width_end = block_width_start + filter_size

            block_height_start = j * stride
            block_height_end = block_height_start + filter_size

            block = X_padded[:, block_width_start:block_width_end, block_height_start:block_height_end, :]

            for d in range(depth):  # Loop over the filters in the layer (output depth dimension)

                filter_weights = W[:, :, :, d]
                # QUESTION: Apply the filter to the block over all inputs in the batch
                # out[:, w, h, f] = ...
                # HINT: Have a look at numpy's sum function and pay attention to the axis parameter
                # ANSWER:
                out[:, i, j, d] = np.sum(block * filter_weights, axis=(1, 2, 3)) + b[d]
                ###

    return out

### Hyperparameters
batch_size = 2
input_width = 4
input_height = 4
input_depth = 3
filter_size = 4
output_depth = 3
stride = 2
padding = 1
###

# Create a helper function that calculates the relative error between two arrays
def relative_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

# Define the shapes of the input and weights
input_shape = (batch_size, input_width, input_height, input_depth)
w_shape = (filter_size, filter_size, input_depth, output_depth)

# Create the dummy input
X = np.linspace(-0.1, 0.5, num=np.prod(input_shape)).reshape(input_shape)

# Create the weights and biases
W = np.linspace(-0.2, 0.3, num=np.prod(w_shape)).reshape(w_shape)
b = np.linspace(-0.1, 0.2, num=output_depth)

# Get the output of the convolutional layer
out = convolutional_forward(X, W, b, filter_size, output_depth, stride, padding)

correct_out = np.array(
        [[[[8.72013250e-02, 2.37300699e-01, 3.87400074e-01],
             [1.34245123e-01, 2.86133235e-01, 4.38021347e-01]],
            [[8.21928598e-02, 2.39447184e-01, 3.96701509e-01],
             [4.47552448e-04, 1.59490615e-01, 3.18533677e-01]]],
         [[[1.11179021e+00, 1.29050939e+00, 1.46922856e+00],
             [9.01255797e-01, 1.08176371e+00, 1.26227162e+00]],
            [[7.64688995e-02, 2.62343025e-01, 4.48217151e-01],
             [-2.62854619e-01, -7.51917556e-02, 1.12471108e-01]]]])

# Compare your output to the "correct" ones
# The difference should be around 2e-8 (or lower)

print('Testing convolutional_forward')
diff = relative_error(out, correct_out)
if diff <= 2e-8:
    print('PASSED')
else:
    print('The difference of x is too high, try again :' + str(diff))


def convolutional_backward(dout, X, W, b, filter_size, depth, stride, padding):
    batch_size, input_width, input_height, input_depth = X.shape

    # Apply padding to the width and height dimensions of the input
    X_padded = np.pad(X, ((0, 0), (padding, padding), (padding, padding), (0, 0)), 'constant')

    # Calculate the width and height of the forward pass output
    output_width = (input_width - filter_size + 2 * padding) / stride + 1
    output_height = (input_height - filter_size + 2 * padding) / stride + 1

    # Allocate output arrays
    # QUESTION: What is the shape of dx? dw? db?
    # ANSWER: ...
    dx_padded = np.zeros_like(X_padded)
    dw = np.zeros_like(W)
    db = np.zeros_like(b)

    # QUESTION: Calculate db, the derivative of the final loss with respect to the bias term
    # HINT: Have a look at the axis parameter of the np.sum function.
    db = np.sum(dout, axis=(0, 1, 2))

    for i in range(output_width):
        for j in range(output_height):

            # Select the current block in the input that the filter will be applied to
            block_width_start = i * stride
            block_width_end = block_width_start + filter_size

            block_height_start = j * stride
            block_height_end = block_height_start + filter_size

            block = X_padded[:, block_width_start:block_width_end, block_height_start:block_height_end, :]

            for d in range(depth):
                # QUESTION: Calculate dw[:,:,:,f], the derivative of the loss with respect to the weight parameters of the f'th filter.
                # HINT: You can do this in a loop if you prefer, or use np.sum and "None" indexing to get your result to the correct
                # shape to assign to dw[:,:,:,f], see (https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html#numpy.newaxis)
                dw[:, :, :, d] += np.sum(block * (dout[:, i, j, d])[:, None, None, None], axis=0)

            dx_padded[:, block_width_start:block_width_end, block_height_start:block_height_end, :] += np.einsum(
                'ij,klmj->iklm', dout[:, i, j, :], W)

        # Now we remove the padding to arrive at dx
        dx = dx_padded[:, padding:-padding, padding:-padding, :]

    return dx, dw, db


def eval_numerical_gradient_array(f, x, df, h=1e-5):
    """
    Evaluate a numeric gradient for a function that accepts a numpy
    array and returns a numpy array.
    """

    # QUESTION: Can you describe intuitively what this function is doing?

    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        oldval = x[ix]
        x[ix] = oldval + h
        pos = f(x).copy()
        x[ix] = oldval - h
        neg = f(x).copy()
        x[ix] = oldval

        grad[ix] = np.sum((pos - neg) * df) / (2 * h)
        it.iternext()
    return grad


np.random.seed(231)

# Normally, backpropagation will have calculated a derivative of the final loss with respect to
# the output of our layer. Since we're testing our layer in isolation here, we'll just pretend
# and use a random value
dout = np.random.randn(2, 2, 2, 3)

dx_num = eval_numerical_gradient_array(
    lambda x: convolutional_forward(X, W, b, filter_size, output_depth, stride, padding), X, dout)
dw_num = eval_numerical_gradient_array(
    lambda w: convolutional_forward(X, W, b, filter_size, output_depth, stride, padding), W, dout)
db_num = eval_numerical_gradient_array(
    lambda b: convolutional_forward(X, W, b, filter_size, output_depth, stride, padding), b, dout)

out = convolutional_forward(X, W, b, filter_size, output_depth, stride, padding)
dx, dw, db = convolutional_backward(dout, X, W, b, filter_size, output_depth, stride, padding)

# Your errors should be around 1e-8'
print('Testing conv_backward_naive function')

dx_diff = relative_error(dx, dx_num)
if dx_diff < 1e-8:
    print('dx check: PASSED')
else:
    print('The difference of a on dx is too high, try again! a: ' + str(dx_diff))

dw_diff = relative_error(dw, dw_num)
if dw_diff < 1e-8:
    print('dw check: PASSED')
else:
    print('The difference of a on dw is too high, try again! a: ' + str(dw_diff))

db_diff = relative_error(db, db_num)
if db_diff < 1e-8:
    print('db check: PASSED')
else:
    print('The difference of a on db is too high, try again! a: ' + str(db_diff))


def max_pool_forward(X, pool_size, stride):
    batch_size, input_width, input_height, input_depth = X.shape

    # Calculate the output dimensions
    output_width = (input_width - pool_size) / stride + 1
    output_height = (input_height - pool_size) / stride + 1

    # Allocate the output array
    out = np.zeros((batch_size, output_width, output_height, input_depth))

    # Select the current block in the input that the filter will be applied to
    for w in range(output_width):
        for h in range(output_height):
            block_width_start = w * stride
            block_width_end = block_width_start + pool_size

            block_height_start = h * stride
            block_height_end = block_height_start + pool_size

            block = X[:, block_width_start:block_width_end, block_height_start:block_height_end, :]
            ## IMPLEMENT-ME CANDIDATE
            out[:, w, h, :] = np.max(block, axis=(1, 2))

    return out


### Hyperparameters
batch_size = 2
input_width = 4
input_height = 4
input_depth = 3
pool_size = 2
stride = 2
###

input_shape = (batch_size, input_width, input_height, input_depth)
X = np.linspace(-0.3, 0.4, num=np.prod(input_shape)).reshape(input_shape)

out = max_pool_forward(X, pool_size, stride)

correct_out = np.array([
        [[[-0.18947368, -0.18210526, -0.17473684],
            [-0.14526316, -0.13789474, -0.13052632]],
         [[-0.01263158, -0.00526316, 0.00210526],
            [0.03157895, 0.03894737, 0.04631579]]],
        [[[0.16421053, 0.17157895, 0.17894737],
            [0.20842105, 0.21578947, 0.22315789]],
         [[0.34105263, 0.34842105, 0.35578947],
            [0.38526316, 0.39263158, 0.4]]]])

# Compare the output. The difference should be less than 1e-6.
print('Testing max_pool_forward function:')
diff = relative_error(out, correct_out)
if diff < 1e-6:
    print('PASSED')
else:
    print('The difference of a is too high, try again! a: ' + str(diff))


def max_pool_backward(dout, X, max_pool_output, pool_size, stride):
    batch_size, input_width, input_height, input_depth = X.shape

    # Calculate the output dimensions
    output_width = (input_width - pool_size) / stride + 1
    output_height = (input_height - pool_size) / stride + 1

    # QUESTION: What is the size of dx, the derivative with respect to x?
    # Allocate an array to hold the derivative
    dx = np.zeros_like(X)

    for w in range(output_width):
        for h in range(output_height):
            # Which block in the input did the value at the forward pass output come from?
            block_width_start = w * stride
            block_width_end = block_width_start + pool_size

            block_height_start = h * stride
            block_height_end = block_height_start + pool_size

            block = X[:, block_width_start:block_width_end, block_height_start:block_height_end, :]

            # What was the maximum value
            max_val = max_pool_output[:, w, h, :]

            # Which values in the input block resulted in the output?
            responsible_values = block == max_val[:, None, None, :]

            # Add the contribution of the current block to the gradient
            dx[:, block_width_start:block_width_end, block_height_start:block_height_end, :] += responsible_values * (dout[:, w, h, :])[:, None, None, :]

    return dx


# Define a hypothetical derivative of the loss function with respect to the output of the max-pooling layer.
dout = np.random.randn(batch_size, pool_size, pool_size, input_depth)

dx_num = eval_numerical_gradient_array(lambda x: max_pool_forward(x, pool_size, stride), X, dout)
out = max_pool_forward(X, pool_size, stride)
dx = max_pool_backward(dout, X, out, pool_size, stride)

# Your error should be less than 1e-12
print('Testing max_pool_backward function:')
diff = relative_error(dx, dx_num)
if diff < 1e-12:
    print('PASSED')
else:
    print('The diff of a is too large, try again! a: ' + str(diff))





################### TF #####################


class BaseSoftmaxClassifier(object):
    def __init__(self, input_size, output_size):
        # Define the input placeholders. The "None" dimension means that the
        # placeholder can take any number of images as the batch size.
        self.x = tf.placeholder(tf.float32, [None, input_size])
        self.y = tf.placeholder(tf.float32, [None, output_size])

        # We add an additional input placeholder for Dropout regularisation
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")

        # And one for bath norm regularisation
        self.is_training = tf.placeholder(tf.bool, name="is_training")

        self.input_size = input_size
        self.output_size = output_size

        # You should override these in your build_model() function.
        self.logits = None
        self.predictions = None
        self.loss = None

        self.build_model()

    def get_logits(self):
        return self.logits

    def build_model(self):
        # OVERRIDE THIS FOR YOUR PARTICULAR MODEL.
        raise NotImplementedError("Subclasses should implement this function!")

    def compute_loss(self):
        """All models share the same softmax cross-entropy loss."""
        assert self.logits is not None  # Ensure that logits has been created!
        data_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y))

        return data_loss

    def accuracy(self):
        # Calculate accuracy.
        assert self.predictions is not None  # Ensure that pred has been created!
        correct_prediction = tf.equal(tf.argmax(self.predictions, 1), tf.argmax(self.y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        return accuracy


def train_tf_model(tf_model,
                   session,  # The active session.
                   num_epochs,  # Max epochs/iterations to train for.
                   batch_size=100,  # Number of examples per batch.
                   keep_prob=1.0,  # (1. - dropout) probability, none by default.
                   optimizer_fn=None,  # TODO(sgouws): more correct to call this optimizer_obj
                   report_every=1,  # Report training results every nr of epochs.
                   eval_every=1,  # Evaluate on validation data every nr of epochs.
                   stop_early=True,  # Use early stopping or not.
                   verbose=True):
    # Get the (symbolic) model input, output, loss and accuracy.
    x, y = tf_model.x, tf_model.y
    loss = tf_model.loss
    accuracy = tf_model.accuracy()

    # Compute the gradient of the loss with respect to the model parameters
    # and create an op that will perform one parameter update using the specific
    # optimizer's update rule in the direction of the gradients.
    if optimizer_fn is None:
        optimizer_fn = tf.train.AdamOptimizer(1e-4)

    # For batch normalisation: Ensure that the mean and variance tracking
    # variables get updated at each training step
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer_step = optimizer_fn.minimize(loss)

    # Get the op which, when executed, will initialize the variables.
    init = tf.global_variables_initializer()
    # Actually initialize the variables (run the op).
    session.run(init)

    # Save the training loss and accuracies on training and validation data.
    train_costs = []
    train_accs = []
    val_costs = []
    val_accs = []

    mnist_train_data = mnist.train

    prev_c_eval = 1000000

    # Main training cycle.
    for epoch in range(num_epochs):

        avg_cost = 0.
        avg_acc = 0.
        total_batch = int(mnist.train.num_examples / batch_size)

        # Loop over all batches.
        for i in range(total_batch):
            batch_x, batch_y = mnist_train_data.next_batch(batch_size)

            # Run optimization op (backprop) and cost op (to get loss value),
            # and compute the accuracy of the model.
            feed_dict = {x: batch_x, y: batch_y, tf_model.keep_prob: keep_prob,
                         tf_model.is_training: True}
            _, c, a = session.run(
                [optimizer_step, loss, accuracy], feed_dict=feed_dict)

            # Compute average loss/accuracy
            avg_cost += c / total_batch
            avg_acc += a / total_batch

        train_costs.append((epoch, avg_cost))
        train_accs.append((epoch, avg_acc))

        # Display logs per epoch step
        if epoch % report_every == 0 and verbose:
            print("Epoch: " + str(epoch + 1) + " Training cost= " + str(avg_cost))

        if epoch % eval_every == 0:
            val_x, val_y = mnist.validation.images, mnist.validation.labels

            feed_dict = {x: val_x, y: val_y, tf_model.keep_prob: 1.0,
                         tf_model.is_training: False}
            c_eval, a_eval = session.run([loss, accuracy], feed_dict=feed_dict)

            if verbose:
                print("Epoch: " + str(epoch + 1) + " Validation acc= " + str(a_eval))

            if c_eval >= prev_c_eval and stop_early:
                print("Validation loss stopped improving, stopping training early after a epochs! a: " + str(epoch + 1))
                break

            prev_c_eval = c_eval

            val_costs.append((epoch, c_eval))
            val_accs.append((epoch, a_eval))

    print("Optimization Finished!")
    return train_costs, train_accs, val_costs, val_accs


def my_plot(list_of_tuples):
    """Take a list of (epoch, value) and split these into lists of
    epoch-only and value-only. Pass these to plot to make sure we
    line up the values at the correct time-steps.
    """
    plt.plot(*zip(*list_of_tuples))


def plot_multi(values_lst, labels_lst, y_label, x_label='epoch'):
    # Plot multiple curves.
    assert len(values_lst) == len(labels_lst)
    plt.subplot(2, 1, 2)

    for v in values_lst:
        my_plot(v)
    plt.legend(labels_lst, loc='upper left')

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def _convolutional_layer(inputs, filter_size, output_depth):
    """Build a convolutional layer with `output_depth` square
    filters, each of size `filter_size` x `filter_size`."""

    input_features = inputs.shape[3]

    weights = tf.get_variable(
        "conv_weights",
        [filter_size, filter_size, input_features, output_depth],
        dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(stddev=0.1))

    ## IMPLEMENT-ME CANDIDATE
    conv = tf.nn.conv2d(inputs, weights, strides=[1, 1, 1, 1], padding='SAME')
    return conv


def _dense_linear_layer(inputs, layer_name, input_size, output_size, weights_initializer):
    """
    Builds a layer that takes a batch of inputs of size `input_size` and returns
    a batch of outputs of size `output_size`.

    Args:
        inputs: A `Tensor` of shape [batch_size, input_size].
        layer_name: A string representing the name of the layer.
        input_size: The size of the inputs
        output_size: The size of the outputs

    Returns:
        out, weights: tuple of layer outputs and weights.

    """
    # Name scopes allow us to logically group together related variables.
    # Setting reuse=False avoids accidental reuse of variables between different runs.
    with tf.variable_scope(layer_name, reuse=False):
        # Create the weights for the layer
        layer_weights = tf.get_variable("weights",
                                        shape=[input_size, output_size],
                                        dtype=tf.float32,
                                        initializer=weights_initializer)
        # Create the biases for the layer
        layer_bias = tf.get_variable("biases",
                                     shape=[output_size],
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.1))

        outputs = tf.matmul(inputs, layer_weights) + layer_bias

    return outputs


class ConvNetClassifier(BaseSoftmaxClassifier):
    def __init__(self,
                 input_size,  # The size of the input
                 output_size,  # The size of the output
                 filter_sizes=[],  # The number of filters to use per convolutional layer
                 output_depths=[],  # The number of features to output per convolutional layer
                 hidden_linear_size=512,  # The size of the hidden linear layer
                 use_batch_norm=False,  # Flag indicating whether or not to use batch normalisation
                 linear_weights_initializer=tf.truncated_normal_initializer(stddev=0.1)):

        assert len(filter_sizes) == len(output_depths)

        self.filter_sizes = filter_sizes
        self.output_depths = output_depths
        self.linear_weights_initializer = linear_weights_initializer
        self.use_batch_norm = use_batch_norm
        self.hidden_linear_size = hidden_linear_size

        super(ConvNetClassifier, self).__init__(input_size, output_size)

    def build_model(self):
        # Architecture: INPUT - {CONV - RELU - POOL}*N - FC

        # Reshape the input to [batch_size, width, height, input_depth]
        conv_input = tf.reshape(self.x, [-1, 28, 28, 1])

        prev_inputs = conv_input

        # Create the CONV-RELU-POOL layers:
        for layer_number, (layer_filter_size, layer_features) in enumerate(
                zip(self.filter_sizes, self.output_depths)):
            with tf.variable_scope("layer_{}".format(layer_number), reuse=False):
                # Create the convolution:
                conv = _convolutional_layer(prev_inputs, layer_filter_size, layer_features)

                # Apply batch normalisation, if required
                if self.use_batch_norm:
                    conv = tf.contrib.layers.batch_norm(conv, center=True, scale=True,
                                                        is_training=self.is_training)

                # Apply the RELU activation with a bias
                bias = tf.get_variable("bias", [layer_features], dtype=tf.float32,
                                       initializer=tf.constant_initializer(0.1))
                relu = tf.nn.relu(conv + bias)

                # Apply max-pooling using patch-sizes of 2x2
                pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1],
                                      strides=[1, 2, 2, 1],
                                      padding='SAME')

                # QUESTION: What is the shape of the pool tensor?
                # ANSWER: ...

                prev_inputs = pool

        # QUESTION: What is the shape of prev_inputs now?

        # We need to flatten the last (non-batch) dimensions of the convolutional
        # output in order to pass it to a fully-connected layer:
        flattened = tf.contrib.layers.flatten(prev_inputs)

        # Create the fully-connected (linear) layer that maps the flattened inputs
        # to `hidden_linear_size` hidden outputs
        flat_size = flattened.shape[1]
        fully_connected = _dense_linear_layer(
            flattened, "fully_connected", flat_size, self.hidden_linear_size, self.linear_weights_initializer)

        # Apply batch normalisation, if required
        if self.use_batch_norm:
            fully_connected = tf.contrib.layers.batch_norm(
                fully_connected, center=True, scale=True, is_training=self.is_training)

        fc_relu = tf.nn.relu(fully_connected)

        fc_drop = tf.nn.dropout(fc_relu, self.keep_prob)

        # Now we map the `hidden_linear_size` outputs to the `output_size` logits, one for each possible digit class
        logits = _dense_linear_layer(
            fc_drop, "logits", self.hidden_linear_size, self.output_size, self.linear_weights_initializer)

        self.logits = logits
        self.predictions = tf.nn.softmax(self.logits)
        self.loss = self.compute_loss()


def build_train_eval_and_plot(build_params, train_params, verbose=True):
    tf.reset_default_graph()
    m = ConvNetClassifier(**build_params)

    with tf.Session() as sess:
        # Train model on the MNIST dataset.

        train_losses, train_accs, val_losses, val_accs = train_tf_model(
            m,
            sess,
            verbose=verbose,
            **train_params)

        # Now evaluate it on the test set:

        accuracy_op = m.accuracy()  # Get the symbolic accuracy operation
        # Calculate the accuracy using the test images and labels.
        accuracy = accuracy_op.eval({m.x: mnist.test.images,
                                     m.y: mnist.test.labels,
                                     m.keep_prob: 1.0,
                                     m.is_training: False})

        if verbose:
            print("Accuracy on test set: " + str(accuracy))
            # Plot losses and accuracies.
            plot_multi([train_losses, val_losses], ['train', 'val'], 'loss', 'epoch')
            plot_multi([train_accs, val_accs], ['train', 'val'], 'accuracy', 'epoch')

        ret = {'train_losses': train_losses, 'train_accs': train_accs,
               'val_losses': val_losses, 'val_accs': val_accs,
               'test_acc': accuracy}

        # Evaluate the final convolutional weights
        conv_variables = [v for v in tf.trainable_variables() if "conv_weights" in v.name]
        conv_weights = sess.run(conv_variables)

        return m, ret, conv_weights


# model_params = {
#         'input_size': 784,
#         'output_size': 10,
#         'filter_sizes': [5],
#         'output_depths': [4],
#         'hidden_linear_size': 128,
#         'use_batch_norm': False
# }
#
# training_params = {
#         'keep_prob': 0.5,
#         'num_epochs': 5,
#         'batch_size': 50,
#         'stop_early': False,
# }

model_params = {
        'input_size': 784,
        'output_size': 10,
        'filter_sizes': [5, 5],
        'output_depths': [32, 64],
        'hidden_linear_size': 1024,
        'use_batch_norm': False
}

training_params = {
        'keep_prob': 0.5,
        'num_epochs': 20,
        'batch_size': 50,
        'stop_early': False,
}

trained_model, training_results, conv_weights = build_train_eval_and_plot(
        model_params,
        training_params,
        verbose=True
)
