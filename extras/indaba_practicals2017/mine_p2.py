import datetime
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

from matplotlib import pyplot as plt
plt.ioff()
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


def display_images(gens, title=""):
    fig, axs = plt.subplots(1, 10, figsize=(25, 3))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    for i in range(10):
        reshaped_img = (gens[i].reshape(28, 28) * 255).astype(np.uint8)
        axs.flat[i].imshow(reshaped_img)
        # axs.flat[i].axis('off')
    return fig, axs


# batch_xs, batch_ys = mnist.train.next_batch(10)
# list_of_images = np.split(batch_xs, 10)
# _ = display_images(list_of_images, "Some Examples from the Training Set.")
# plt.show()


# Build model
def _dense_linear_layer(inputs, layer_name, input_size, output_size):
    with tf.variable_scope(layer_name, reuse=False):
        layer_weights = tf.get_variable('weights', shape=[input_size, output_size], dtype=tf.float32,
                                        initializer=tf.random_normal_initializer())
        layer_bias = tf.get_variable('biases', shape=[output_size], dtype=tf.float32,
                                     initializer=tf.random_normal_initializer())

        outputs = tf.matmul(inputs, layer_weights) + layer_bias
    return (outputs, layer_weights)


class BaseSoftmaxClassifier(object):
    def __init__(self, input_size, output_size, l2_lambda):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, input_size])
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, output_size])
        self.input_size = input_size
        self.output_size = output_size
        self.l2_lambda = l2_lambda

        self._all_weights = []
        self.logits = None
        self.predictions = None
        self.loss = None

        self.build_model()

    def get_logits(self):
        return self.logits

    def build_model(self):
        raise NotImplementedError("Subclasses should implement this function!")

    def compute_loss(self):
        assert self.logits is not None
        data_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y))

        reg_loss = 0.0;
        for w in self._all_weights:
            reg_loss += tf.nn.l2_loss(w)

        return data_loss + (self.l2_lambda * reg_loss)

    def accuracy(self):
        assert self.predictions is not None
        correct_prediction = tf.equal(tf.argmax(self.predictions, axis=1), tf.argmax(self.y, axis=1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
        return accuracy


class LinearSoftmaxClassifier(BaseSoftmaxClassifier):
    def __init__(self, input_size, output_size, l2_lambda):
        super(LinearSoftmaxClassifier, self).__init__(input_size, output_size, l2_lambda)

    def build_model(self):
        self.logits, weights = _dense_linear_layer(self.x, 'linear_layer', self.input_size, self.output_size)
        self._all_weights.append(weights)
        self.predictions = tf.nn.softmax(self.logits)
        self.loss = self.compute_loss()


class DNNClassifier(BaseSoftmaxClassifier):
    def __init__(self, input_size=784, hidden_sizes=[], output_size=10, act_fn=tf.nn.relu, l2_lambda=0.0):
        self.hidden_sizes = hidden_sizes
        self.act_fn = act_fn
        super(DNNClassifier, self).__init__(input_size, output_size, l2_lambda)

    def build_model(self):
        prev_layer = self.x
        prev_size = self.input_size

        for layer_num, size in enumerate(self.hidden_sizes):
            layer_name = "layer_" + str(layer_num)
            layer, weights = _dense_linear_layer(prev_layer, layer_name, prev_size, size)
            self._all_weights.append(weights)
            layer = self.act_fn(layer)
            prev_layer, prev_size = layer, size

        self.logits, out_weights = _dense_linear_layer(prev_layer, 'output', prev_size, self.output_size)
        self._all_weights.append(out_weights)
        self.predictions = tf.nn.softmax(self.logits)
        self.loss = self.compute_loss()


tf.set_random_seed(1234)
np.random.seed(1234)

# Generate a batch of 100 "images" of 784 pixels consisting of Gaussian noise.
x_rnd = np.random.randn(100, 784)
print("Sample of random data:\n" + str(x_rnd[:5,:]))    # Print the first 5 "images"
print("Shape: " + str(x_rnd.shape))
# Generate some random one-hot labels.
y_rnd = np.eye(10)[np.random.choice(10, 100)]
print("Sample of random labels:\n" + str(y_rnd[:5,:]))
print("Shape: " + str(y_rnd.shape))

# No reg
tf.reset_default_graph()
tf_linear_model = DNNClassifier(l2_lambda=0.0)
x, y = tf_linear_model.x, tf_linear_model.y

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    avg_cross_entropy = -tf.log(tf.reduce_mean(tf_linear_model.predictions))
    loss_no_reg = tf_linear_model.loss
    manual_avg_xent, loss_no_reg = sess.run([avg_cross_entropy, loss_no_reg], feed_dict={x: x_rnd, y: y_rnd})

# Sanity check: Loss should be about log(10) = 2.3026
print('\nSanity check manual avg cross entropy: ' + str(manual_avg_xent))
print('Model loss (no reg): ' + str(loss_no_reg))

# With reg
tf.reset_default_graph()
tf_linear_model = DNNClassifier(l2_lambda=1.0)
x, y = tf_linear_model.x, tf_linear_model.y

with tf.Session() as sess:
    # Initialize variables.
    init = tf.global_variables_initializer()
    sess.run(init)
    loss_w_reg = tf_linear_model.loss.eval(feed_dict={x : x_rnd, y: y_rnd})

# Sanity check: Loss should go up when you add regularization
print('Sanity check loss (with regularization, should be higher): ' + str(loss_w_reg))


class MNISTFraction(object):
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

        # Loop over all batches.
        for i in range(total_batch):
            batch_x, batch_y = mnist_train_data.next_batch(batch_size)

            feed_dict = {x: batch_x, y: batch_y}
            if keep_prob < 1.:
                feed_dict["keep_prob:0"] = keep_prob

            _, c, a = session.run(
                [optimizer_step, loss, accuracy], feed_dict=feed_dict)

            avg_cost += c / total_batch
            avg_acc += a / total_batch

        train_costs.append((epoch, avg_cost))
        train_accs.append((epoch, avg_acc))

        if epoch % report_every == 0 and verbose:
            print("Epoch: " + str(epoch + 1) + " Training cost= " + str(avg_cost))

        if epoch % eval_every == 0:
            val_x, val_y = mnist.validation.images, mnist.validation.labels

            feed_dict = {x: val_x, y: val_y}
            if keep_prob < 1.:
                feed_dict['keep_prob:0'] = 1.0

            c_eval, a_eval = session.run([loss, accuracy], feed_dict=feed_dict)

            if verbose:
                print("Epoch: " + str(epoch + 1) + " Validation acc= " + str(a_eval))

            if c_eval >= prev_c_eval and stop_early:
                print("Validation loss stopped improving, stopping training early after x epochs! :" + str(epoch + 1))
                break

            prev_c_eval = c_eval

            val_costs.append((epoch, c_eval))
            val_accs.append((epoch, a_eval))

    print("Optimization Finished!")
    return train_costs, train_accs, val_costs, val_accs


def my_plot(list_of_tuples):
    plt.plot(*zip(*list_of_tuples))


def plot_multi(values_lst, labels_lst, y_label, x_label='epoch'):
    assert len(values_lst) == len(labels_lst)
    plt.subplot(2, 1, 2)

    for v in values_lst:
        my_plot(v)
    plt.legend(labels_lst, loc='upper left')

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


tf.reset_default_graph()  # Clear the graph.
model = DNNClassifier()  # Choose model hyperparameters.

with tf.Session() as sess:
    train_losses, train_accs, val_losses, val_accs = train_tf_model(
        model,
        session=sess,
        num_epochs=10,
        train_only_on_fraction=1e-1,
        optimizer_fn=tf.train.GradientDescentOptimizer(learning_rate=1e-3),
        report_every=1,
        eval_every=2,
        stop_early=False)

    accuracy_op = model.accuracy()  # Get the symbolic accuracy operation
    accuracy = accuracy_op.eval(feed_dict={model.x: mnist.test.images,
                                           model.y: mnist.test.labels})

    print("Accuracy on test set: " + str(accuracy))


def build_train_eval_and_plot(build_params, train_params, verbose=True):
    tf.reset_default_graph()
    m = DNNClassifier(**build_params)

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
# Specify the model hyperparameters (NOTE: All the defaults can be omitted):
model_params = {
    # 'input_size' : 784,    # There are 28x28 = 784 pixels in MNIST images
    'hidden_sizes': [512],  # List of hidden layer dimensions, empty for linear model.
    # 'output_size' : 10,    # There are 10 possible digit classes
    # 'act_fn' : tf.nn.relu,    # The activation function to use in the hidden layers
    'l2_lambda': 0.  # Strength of L2 regularization.
}

# Specify the training hyperparameters:
training_params = {'num_epochs': 100,  # Max epochs/iterations to train for.
                   # 'batch_size' : 100,    # Number of examples per batch, 100 default.
                   # 'keep_prob' : 1.0,    # (1. - dropout) probability, none by default.
                   'train_only_on_fraction': 5e-2,  # Fraction of training data to use, 1. for everything.
                   'optimizer_fn': None,  # Optimizer, None for Adam.
                   'report_every': 1,  # Report training results every nr of epochs.
                   'eval_every': 2,  # Evaluate on validation data every nr of epochs.
                   'stop_early': False,  # Use early stopping or not.
                   }

# Build, train, evaluate and plot the results!
# trained_model, training_results = build_train_eval_and_plot(
#     model_params,
#     training_params,
#     verbose=True  # Modify as desired.
# )

###############################END CODE TEMPLATE################################


# Train the linear model on the full dataset.
################################################################################
# Specify the model hyperparameters.
model_params = {'l2_lambda' : 0.}

# Specify the training hyperparameters:
training_params = {'num_epochs' : 50,     # Max epochs/iterations to train for.
                        'optimizer_fn' : None,            # Now we're using Adam.
                        'report_every' : 1, # Report training results every nr of epochs.
                        'eval_every' : 1,     # Evaluate on validation data every nr of epochs.
                        'stop_early' : True
}

# Build, train, evaluate and plot the results!
# trained_model, training_results = build_train_eval_and_plot(
#         model_params,
#         training_params,
#         verbose=True    # Modify as desired.
# )
################################################################################


# Specify the model hyperparameters (NOTE: All the defaults can be omitted):
model_params = {
        'hidden_sizes' : [512],    # List of hidden layer dimensions, empty for linear model.
        'l2_lambda' : 1e-3            # Strength of L2 regularization.
}

# Specify the training hyperparameters:
training_params = {
        'num_epochs' : 50,        # Max epochs/iterations to train for.
        'report_every' : 1,     # Report training results every nr of epochs.
        'eval_every' : 1,         # Evaluate on validation data every nr of epochs.
        'stop_early' : True    # Use early stopping or not.
}

# Build, train, evaluate and plot the results!
# trained_model, training_results = build_train_eval_and_plot(
#         model_params,
#         training_params,
#         verbose=True    # Modify as desired.
# )

################################################################################ fdgdf

# Specify the model hyperparameters (NOTE: All the defaults can be omitted):
model_params = {
        'hidden_sizes' : [512, 512], # List of hidden layer dimensions, empty for linear model.
        'l2_lambda' : 1e-3                     # Strength of L2 regularization.
}

# Specify the training hyperparameters:
training_params = {
        'num_epochs' : 200,     # Max epochs/iterations to train for.
        'report_every' : 1,     # Report training results every nr of epochs.
        'eval_every' : 1,         # Evaluate on validation data every nr of epochs.
        'stop_early' : True,    # Use early stopping or not.
}

# Build, train, evaluate and plot the results!
# trained_model, training_results = build_train_eval_and_plot(
#         model_params,
#         training_params,
#         verbose=True    # Modify as desired.
# )

################################################################################

# Best

# Specify the model hyperparameters (NOTE: All the defaults can be omitted):
model_params = {
        'hidden_sizes' : [500, 300], # List of hidden layer dimensions, empty for linear model.
        'l2_lambda' : 1e-3                     # Strength of L2 regularization.
}

# Specify the training hyperparameters:
training_params = {'num_epochs' : 100,     # Max epochs/iterations to train for.
                        'optimizer_fn' : tf.train.MomentumOptimizer(learning_rate=2e-3, momentum=0.98),
                        'report_every' : 1, # Report training results every nr of epochs.
                        'eval_every' : 1,     # Evaluate on validation data every nr of epochs.
                        'stop_early' : True,    # Use early stopping or not.
}

# Build, train, evaluate and plot the results!
trained_model, training_results = build_train_eval_and_plot(
        model_params,
        training_params,
        verbose=True    # Modify as desired.
)
