# RNN model
import tensorflow as tf
import numpy as np
import functools
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# Reuse softmax classofier
class BaseSoftmaxClassifier(object):
    def __init__(self, input_size, output_size, l2_lambda):
        # Define the input placeholders. The "None" dimension means that the
        # placeholder can take any number of images as the batch size.
        self.x = tf.placeholder(tf.float32, [None, input_size], name='x')
        self.y = tf.placeholder(tf.float32, [None, output_size], name='y')
        self.input_size = input_size
        self.output_size = output_size
        self.l2_lambda = l2_lambda

        self._all_weights = []  # Used to compute L2 regularization in compute_loss().

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
        reg_loss = 0.
        for w in self._all_weights:
            reg_loss += tf.nn.l2_loss(w)

        return data_loss + self.l2_lambda * reg_loss

    def accuracy(self):
        # Calculate accuracy.
        assert self.predictions is not None  # Ensure that pred has been created!
        correct_prediction = tf.equal(tf.argmax(self.predictions, 1), tf.argmax(self.y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        return accuracy


# Recurrent Classifier
class RecurrentClassifier(BaseSoftmaxClassifier):
    def __init__(self, model_params):
        self.config = model_params

        super(RecurrentClassifier, self).__init__(model_params['input_size'],
                                                  model_params['output_size'],
                                                  model_params['l2_lambda'])

    def build_model(self):
        assert self.config['num_steps'] * self.config['pixels_per_step'] == self.config['input_size']
        # We break up the input images into num_steps groups of pixels_per_step
        # pixels each.
        rnn_input = tf.reshape(self.x, [-1,
                                        self.config['num_steps'],
                                        self.config['pixels_per_step']])

        # Define the main RNN 'cell', that will be applied to each timestep.
        cell = self.config['cell_fn'](self.config['memory_units'])
        # NOTE: This is how we apply Dropout to RNNs.
        cell = tf.contrib.rnn.DropoutWrapper(
            cell,
            output_keep_prob=self.config['dropout_keep_prob'])
        cell = tf.contrib.rnn.MultiRNNCell(cells=[cell] * self.config['num_layers'],
                                           state_is_tuple=True)
        ##########

        outputs, state = tf.nn.dynamic_rnn(cell,
                                           rnn_input,
                                           dtype=tf.float32)

        # Transpose the cell to get the output from the last timestep for each batch.
        output = tf.transpose(outputs, [1, 0, 2])
        last_hiddens = tf.gather(output, int(output.get_shape()[0]) - 1)

        # Define weights and biases for output prediction.
        out_weights = tf.Variable(tf.random_normal([self.config['memory_units'],
                                                    self.config['output_size']]))
        self._all_weights.append(out_weights)

        out_biases = tf.Variable(tf.random_normal([self.config['output_size']]))

        self.logits = tf.matmul(last_hiddens, out_weights) + out_biases

        self.predictions = tf.nn.softmax(self.logits)  # sigmoid ?
        self.loss = self.compute_loss()

    def get_logits(self):
        return self.logits


# MNIST fraction
class MNISTFraction(object):
    """A helper class to extract only a fixed fraction of MNIST data."""

    def __init__(self, mnist, fraction):
        self.mnist = mnist
        self.num_images = int(mnist.num_examples * fraction)
        self.image_data, self.label_data = mnist.images[:self.num_images], mnist.labels[:self.num_images]
        self.start = 0

    def next_batch(self, batch_size):
        start = self.start
        end = min(start + batch_size, self.num_images)
        self.start = 0 if end == self.num_images else end
        return self.image_data[start:end], self.label_data[start:end]


# Train model
def train_tf_model(tf_model,
                   session,  # The active session.
                   num_epochs,  # Max epochs/iterations to train for.
                   batch_size=50,  # Number of examples per batch.
                   keep_prob=1.0,  # (1. - dropout) probability, none by default.
                   train_only_on_fraction=1.,  # Fraction of training data to use.
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
        optimizer_fn = tf.train.AdamOptimizer()
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

    if train_only_on_fraction < 1:
        mnist_train_data = MNISTFraction(mnist.train, train_only_on_fraction)
    else:
        mnist_train_data = mnist.train

    prev_c_eval = 1000000

    # Main training cycle.
    for epoch in range(num_epochs):

        avg_cost = 0.
        avg_acc = 0.
        total_batch = int(train_only_on_fraction * mnist.train.num_examples / batch_size)

        ## IMPLEMENT-ME: ...
        # Loop over all batches.
        for i in range(total_batch):
            batch_x, batch_y = mnist_train_data.next_batch(batch_size)

            # Run optimization op (backprop) and cost op (to get loss value),
            # and compute the accuracy of the model.
            feed_dict = {x: batch_x, y: batch_y}
            if keep_prob < 1.:
                feed_dict["keep_prob:0"] = keep_prob

            _, c, a = session.run(
                [optimizer_step, loss, accuracy], feed_dict=feed_dict)

            # Compute average loss/accuracy
            avg_cost += c / total_batch
            avg_acc += a / total_batch

        train_costs.append((epoch, avg_cost))
        train_accs.append((epoch, avg_acc))

        # Display logs per epoch step
        if epoch % report_every == 0 and verbose:
            print("Epoch: " + str(epoch + 1) + " Training cost = " + str(avg_cost))

        if epoch % eval_every == 0:
            val_x, val_y = mnist.validation.images, mnist.validation.labels

            feed_dict = {x: val_x, y: val_y}
            if keep_prob < 1.:
                feed_dict['keep_prob:0'] = 1.0

            c_eval, a_eval = session.run([loss, accuracy], feed_dict=feed_dict)

            if verbose:
                print("Epoch: " + str(epoch + 1) + " Validation acc= " + str(a_eval))

            if c_eval >= prev_c_eval and stop_early:
                print(
                "Validation loss stopped improving, stopping training early after x epochs! x = " + str(epoch + 1))
                break

            prev_c_eval = c_eval

            val_costs.append((epoch, c_eval))
            val_accs.append((epoch, a_eval))

    print("Optimization Finished!")
    return train_costs, train_accs, val_costs, val_accs


# Plotting
from matplotlib import pyplot as plt

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


# Train model
def build_train_eval_and_plot(model_params, train_params, verbose=True):
    tf.reset_default_graph()
    m = RecurrentClassifier(model_params)

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
                                     m.y: mnist.test.labels})

        if verbose:
            print("Accuracy on test set: " + str(accuracy))
            # Plot losses and accuracies.
            plot_multi([train_losses, val_losses], ['train', 'val'], 'loss', 'epoch')
            plot_multi([train_accs, val_accs], ['train', 'val'], 'accuracy', 'epoch')

        ret = {'train_losses': train_losses, 'train_accs': train_accs,
               'val_losses': val_losses, 'val_accs': val_accs,
               'test_acc': accuracy}

        return m, ret




#################################CODE TEMPLATE##################################
# Specify the model hyperparameters:
model_params = {
    'input_size': 784,
    'output_size': 10,  # 10
    'batch_size': 100,
    'num_steps': 28,
    'pixels_per_step': 28,  # NOTE: num_steps * pixels_per_step must = input_size
    'cell_fn': tf.contrib.rnn.BasicRNNCell,
    'memory_units': 256,
    'num_layers': 1,
    'l2_lambda': 1e-3,
    'dropout_keep_prob': 1.0
}

# Specify the training hyperparameters:
training_params = {
    'num_epochs': 1,  # Max epochs/iterations to train for. default = 100
    'batch_size': 100,  # Number of examples per batch, 100 default.
    # 'keep_prob' : 1.0,    # (1. - dropout) probability, none by default.
    'train_only_on_fraction': 1.0,  # Fraction of training data to use, 1. for everything.
    'optimizer_fn': None,  # Optimizer, None for Adam.
    'report_every': 1,  # Report training results every nr of epochs.
    'eval_every': 1,  # Evaluate on validation data every nr of epochs.
    'stop_early': True,  # Use early stopping or not.
}

# Build, train, evaluate and plot the results!
trained_model, training_results = build_train_eval_and_plot(
    model_params,
    training_params,
    verbose=True  # Modify as desired.
)

###############################END CODE TEMPLATE################################
