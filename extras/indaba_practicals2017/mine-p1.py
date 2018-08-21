import numpy as np
import matplotlib.pyplot as plt

center = 1.0
points_in_class = 20
np.random.seed(0)

x_pos = np.random.normal(loc=center, scale=1.0, size=[points_in_class, 2])
x_neg = np.random.normal(loc=-center, scale=1.0, size=[points_in_class, 2])
x = np.concatenate((x_pos, x_neg), axis=0)

y_pos = np.ones(points_in_class)
y_neg = - np.ones(points_in_class)
y = np.concatenate((y_pos, y_neg), axis=0)

N = 2 * points_in_class


W = [2.0, 2.0] # change these to see impact
t = 2.0

# fig = plt.figure()
# plt.scatter(x[:, 0], x[:, 1], c=y, s=N)
# plt.axis('equal')

# plt.plot([0, W[0]], [0, W[1]], 'k-')
# plt.plot([-t * W[1], t * W[1]], [t * W[0], -t * W[0]], 'r-')

# plt.xlabel('x0')
# plt.ylabel('x1')
# plt.show()


def compute_loss(w0, w1, x, y, alpha):
    loss = alpha * (w0 * w0 + w1 * w1)

    for n in range(N):
        inner = w0 * x[n, 0] + w1 * x[n, 1]
        loss += np.log(1 + np.exp(- y[n] * inner))

    return loss

lim = 5
ind = np.linspace(-lim, lim, 50)
w0, w1 = np.meshgrid(ind, ind)

alpha = 0.1
loss = compute_loss(w0, w1, x, y, alpha)

# fig = plt.figure()
# plt.contourf(w0, w1, np.exp(-loss), 20, cmap=plt.cm.jet)
# cbar = plt.colorbar()

# plt.title('A plot of exp(-loss), as a function of weight vector [w0, w1]; '
#                     + 'alpha = ' + str(alpha))
# plt.xlabel('w0')
# plt.ylabel('w1')
# plt.axis('equal')
# plt.show()

def reset_matplotlib():
    # %matplotlib inline
    plt.rcParams['figure.figsize'] = (10.0, 8.0)
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'

reset_matplotlib()
# %load_ext autoreload
# %autoreload 2

# Hyper params
num_classes = 3
dimensions = 2
points_per_class = 100

np.random.seed(0)

def generate_spiral_data(num_classes, dimensions, points_per_class):
    """ test """
    X = np.zeros((points_per_class*num_classes, dimensions), dtype='float32')
    y = np.zeros(points_per_class*num_classes, dtype='uint8')

    for y_value in range(num_classes):
        ix = range(points_per_class * y_value, points_per_class * (y_value + 1))

        radius = np.linspace(0.0, 1, points_per_class)
        theta = np.linspace(y_value * 4, (y_value + 1) * 4, points_per_class) + np.random.randn(points_per_class) * 0.2

        X[ix] = np.column_stack([radius * np.sin(theta), radius * np.cos(theta)])
        y[ix] = y_value
    
    return X, y

def plot_data(X, y):
    fig = plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    return fig

X, y = generate_spiral_data(num_classes, dimensions, points_per_class)
fig = plot_data(X, y)
# plt.show()

idx = np.random.choice(range(y.size), size=10, replace=False)
print("X values: " + str(X[idx,]))
print('')
print("Y values: " + str(y[idx,]))

# Classifier from scratch
learning_rate = 1e-0    # "Step-size": How far along the gradient do we want to
reg_lambda = 1e-3    # Regularization strength.
W_init = 0.01 * np.random.randn(dimensions, num_classes)


def softmax(logits):
    exp_logits = np.exp(logits)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    return probs


def cross_entropy(predictions, targets):
    num_examples = predictions.shape[0]
    correct_logprobs = -np.log(predictions[range(num_examples), targets])
    crossentropy = np.sum(correct_logprobs) / num_examples
    return crossentropy

def l2_loss(parameters):
    reg = 0.0
    for param in parameters:
        reg += 0.5 * reg_lambda * np.sum(param * param)
    return reg


class LinearModel(object):
    def __init__(self):
        # Initialize the model parameters.
        self.W = np.copy(W_init)
        self.b = np.zeros((1, num_classes))

    def predictions(self, X):
        """Make predictions of classes (y values) given some inputs (X)."""
        # Evaluate class scores/"logits": [points_per_class*num_classes x num_classes].
        logits = self.get_logits(X)

        # Compute the class probabilities.
        probs = softmax(logits)

        return probs

    def loss(self, probs, y):
        """Calculate the loss given model predictions and true targets."""
        num_examples = probs.shape[0]
        data_loss = cross_entropy(probs, y)
        regulariser = l2_loss([self.W])
        return data_loss + regulariser

    def update(self, probs, X, y):
        """Update the model parameters using back-propagation and gradient descent."""
        # Calculate the gradient of the loss with respect to logits
        dlogits = self.derivative_loss_logits(probs, y)

        # Gradient of the loss wrt W
        dW = self.derivative_loss_W(X, dlogits)

        # Gradient of the loss wrt b
        db = self.derivative_loss_b(dlogits)

        # Don't forget the gradient on the regularization term.
        dW += self.derivative_regularisation()

        # Perform a parameter update.
        self.W += -learning_rate * dW
        self.b += -learning_rate * db

    def get_logits(self, X):
        """Calculate the un-normalised model scores."""
        return np.dot(X, self.W) + self.b

    def derivative_loss_logits(self, probs, y):
        """Calculate the derivative of the loss with respect to logits."""
        num_examples = y.shape[0]
        dlogits = probs
        dlogits[range(num_examples), y] -= 1
        dlogits /= num_examples
        return dlogits

    def derivative_loss_W(self, X, dlogits):
        """Calculate the derivative of the loss wrt W."""
        return np.dot(X.T, dlogits)

    def derivative_loss_b(self, dlogits):
        """Calculate the derivative of the loss wrt b."""
        return np.sum(dlogits, axis=0, keepdims=True)

    def derivative_regularisation(self):
        return reg_lambda * self.W

# Train the model
def train_model(model, epochs, report_every, render_fn=None, render_args={}):
    frames = []
    for i in range(epochs):
        probs = model.predictions(X)
        loss = model.loss(probs, y)

        # Print the loss value every `report_every` steps.
        if i % report_every == 0:
            print("iteration: " + str(i) + " loss: " + str(loss))
            if render_fn:
                frame = render_fn(**render_args)
                frames.append(frame)

        model.update(probs, X, y)
    if frames: return frames

# Create and train model
linear_model = LinearModel()
train_model(linear_model, 200, 10)


# Evaluate the model
def evaluate_model(model):
    scores = model.get_logits(X)
    predicted_class = np.argmax(scores, axis=1)
    print('Accuracy: ' + str(np.mean(predicted_class == y)))

evaluate_model(linear_model)


# Define a function that plots the decision boundary of a model
def plot_decision_boundary(X, model, render=True):
    step_size = 0.02  # Discretization step-size

    # Get the boundaries of the dataset.
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    # Generate a grid of points, step_size apart, over the above region.
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step_size),
                         np.arange(y_min, y_max, step_size))

    # Flatten the data and get the logits of the classifier (the "scores") for
    #     each point in the generated mesh-grid.
    meshgrid_matrix = np.c_[xx.ravel(), yy.ravel()]
    Z = model.get_logits(meshgrid_matrix)

    # Get the class predictions for each point.
    Z = np.argmax(Z, axis=1)
    Z = Z.reshape(xx.shape)

    old_backend = plt.rcParams['backend']  # Save backend.
    if not render:
        plt.rcParams['backend'] = 'agg'

    # Overlay both of these on one figure.
    fig = plt.figure()
    axes = plt.gca()

    axes.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    axes.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    plt.show()

    if not render:
        # Now we can save it to a numpy array.
        fig.tight_layout(pad=0)
        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        # Restore old backend
        plt.rcParams['backend'] = old_backend

        return data

        # fig.savefig('spiral_linear.png')

from matplotlib import animation
from IPython.display import display
from IPython.display import HTML

def display_frames_as_gif(frames):
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi = 72)
    patch = plt.imshow(frames[0])
    #plt.axis('off')

    def animate(i):
            patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    ##display(display_animation(anim, default_mode='loop'))
    HTML(anim.to_html5_video())
    # METHOD 2
    #plt.rcParams['animation.html'] = 'html5'
    #anim
    return anim

# Create an instance of our LinearModel.
reset_matplotlib()
linear_model = LinearModel()

train_model(linear_model, 200, 10)
# plot_decision_boundary(X, linear_model)

# Non Linear
def relu(value):
    return np.maximum(0, value)

learning_rate = 1e-0
reg_lambda = 1e-3
num_hidden = 100
non_linear_W_init = 0.01 * np.random.rand(dimensions, num_hidden)
non_linear_W2_init = 0.01 * np.random.rand(num_hidden, num_classes)

# Non Lin Model
class NonLinearModel(object):
    def __init__(self):
        self.W = non_linear_W_init
        self.b = np.zeros((1, num_hidden))
        self.W2 = non_linear_W2_init
        self.b2 = np.zeros((1, num_classes))

    def predictions(self, X):
        logits = self.get_logits(X)

        probs = softmax(logits)

        return probs

    def loss(self, probs, y):
        data_loss = cross_entropy(probs, y)
        regulariser = l2_loss([self.W, self.W2])
        return data_loss + regulariser

    def update(self, probs, X, y):
        hidden_output = self.hidden_layer(X)

        dlogits = self.derivative_loss_logits(probs, y)

        dW2 = self.derivative_loss_W2(hidden_output, dlogits)
        db2 = self.derivative_loss_b2(dlogits)

        # Next, backprop into the hidden layer.
        dhidden = self.derivative_hidden(hidden_output, dlogits)

        dW = self.derivative_loss_W(X, dhidden)
        db = self.derivative_loss_b(dhidden)

        dW2 += self.derivative_regularisation_W2()
        dW += self.derivative_regularisation_W()

        self.W += -learning_rate * dW
        self.b += -learning_rate * db
        self.W2 += -learning_rate * dW2
        self.b2 += -learning_rate * db2

    ## DEFINE THE MODEL HELPER FUNCTIONS

    def hidden_layer(self, X):
        return relu(np.dot(X, self.W) + self.b)

    def get_logits(self, X):
        hidden_output = self.hidden_layer(X)
        logits = np.dot(hidden_output, self.W2) + self.b2
        return logits

    def derivative_loss_logits(self, logits, y):
        num_examples = y.shape[0]
        dlogits = logits
        dlogits[range(num_examples), y] -= 1
        dlogits /= num_examples
        return dlogits

    def derivative_loss_W2(self, hidden_output, dlogits):
        dW2 = np.dot(hidden_output.T, dlogits)
        return dW2

    def derivative_loss_b2(self, dlogits):
        return np.sum(dlogits, axis=0, keepdims=True)

    def derivative_hidden(self, hidden_output, dlogits):
        dhidden = np.dot(dlogits, self.W2.T)
        dhidden[hidden_output <= 0] = 0

        return dhidden

    def derivative_loss_W(self, X, dhidden):
        return np.dot(X.T, dhidden)

    def derivative_loss_b(self, dhidden):
        return np.sum(dhidden, axis=0, keepdims=True)

    def derivative_regularisation_W(self):
        return reg_lambda * self.W

    def derivative_regularisation_W2(self):
        return reg_lambda * self.W2


# call non lin model
non_linear_model = NonLinearModel()
train_model(non_linear_model, 10000, 1000)
evaluate_model(non_linear_model)
# plot_decision_boundary(X, non_linear_model)

# Tensorflow
import tensorflow as tf
tf.reset_default_graph() # Clear the graph between different invocations.
sess = tf.InteractiveSession() # Create and register the default Session.

### HYPERPARAMETERS
learning_rate = 1e-0
reg_lambda = 1e-3
training_iterations = 200    # 'epochs'
batch_size = X.shape[0]    # The whole dataset; i.e. batch gradient descent.
display_step = 10    # How often should we print our results
###

# Network Parameters
num_input = 2 # 2-dimensional input data
num_classes = 3 # red, yellow and blue!

x_tf = tf.placeholder(tf.float32, [None, num_input])
y_tf = tf.placeholder(tf.int32, [None])

def cross_entropy_tf(predictions, targets):
    targets = tf.one_hot(targets, num_classes)
    return tf.reduce_mean(-tf.reduce_sum(targets * tf.log(predictions), axis=1))

# TF linear model
class TFLinearModel(object):
    def __init__(self):
        # Initialise the variables
        # Tensorflow variables can be updated automatically by optimisers.
        self.W = tf.Variable(W_init, dtype=tf.float32)
        self.b = tf.Variable(tf.zeros([num_classes]), dtype=tf.float32)

    def predictions(self, X):
        """Make predictions of classes (y values) given some inputs (X)."""
        logits = self.get_logits(X)

        # Compute the class probabilities.
        probs = tf.nn.softmax(logits)

        return probs

    def loss(self, probs, y):
        """Calculate the loss given model predictions and true targets."""
        data_loss = cross_entropy_tf(probs, y)
        regulariser = reg_lambda * tf.nn.l2_loss(self.W)
        return data_loss + regulariser

    def get_logits(self, X):
        # An affine function.
        return tf.add(tf.matmul(X, self.W), self.b)


def train_tf_model(tf_model, epochs, report_every):
    init = tf.global_variables_initializer()
    probs = tf_model.predictions(x_tf)
    loss = tf_model.loss(probs, y_tf)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

    optimizer_step = optimizer.minimize(loss)

    sess.run(init)

    # Training cycle.
    for iteration in range(epochs):
        avg_cost = 0.
        total_batch = int(X.shape[0] / batch_size)

        # Loop over all batches.
        for i in range(total_batch):
            batch_x = X[i * batch_size: (i + 1) * batch_size, :]
            batch_y = y[i * batch_size: (i + 1) * batch_size]

            # Run optimization op (backprop) and cost op (to get loss value).
            _, c = sess.run([optimizer_step, loss], feed_dict={x_tf: batch_x, y_tf: batch_y})
            # Compute average loss.
            avg_cost += c / total_batch

        # Display logs per iteration/epoch step.
        if iteration % report_every == 0:
            print("Iteration: " + str(iteration + 1) + " cost: " + str(avg_cost))

    print("Optimization Finished!")

# Intialize our TensorFlow Linear Model
tf_linear_model = TFLinearModel()
train_tf_model(tf_linear_model, 200, 10)


# Vis TF lin model
class TFModelWrapper(object):
    def __init__(self, model):
        self._model = model

    def get_logits(self, x):
        return tf.get_default_session().run(self._model.get_logits(x_tf),
                                            feed_dict={x_tf: x,
                                                       y_tf: np.zeros(x.shape[0])})

wrapper = TFModelWrapper(tf_linear_model)
# plot_decision_boundary(X, wrapper)

# NON LINEAR CLASSIFIER
class TFNonLinearModel(object):
    def __init__(self):
        self.W = tf.Variable(non_linear_W_init, dtype=tf.float32)
        self.b = tf.Variable(tf.zeros([num_hidden]), dtype=tf.float32)
        self.W2 = tf.Variable(non_linear_W2_init, dtype=tf.float32)
        self.b2 = tf.Variable(tf.zeros([num_classes]), dtype=tf.float32)

    def predictions(self, X):
        logits = self.get_logits(X)
        probs = tf.nn.softmax(logits)
        return probs

    def loss(self, probs, y):
        data_loss = cross_entropy_tf(probs, y)
        regulariser = reg_lambda * tf.nn.l2_loss(self.W) + reg_lambda * tf.nn.l2_loss(self.W2)
        # regulariser = 0.0  # reg_lambda * tf.nn.l2_loss(self.W) + reg_lambda * tf.nn.l2_loss(self.W2)
        return data_loss + regulariser

    def get_logits(self, X):
        hidden_output = self.hidden_layer(X)
        logits = tf.add(tf.matmul(hidden_output, self.W2), self.b2)
        return logits

    def hidden_layer(self, X):
        linear = tf.add(tf.matmul(X, self.W), self.b)
        return tf.nn.relu(linear)

tf_non_linear_model = TFNonLinearModel()
train_tf_model(tf_non_linear_model, 10000, 1000)

wrapper = TFModelWrapper(tf_non_linear_model)
plot_decision_boundary(X, wrapper)

# Reg xperiment
# for v in [0., 1e-4, 1e-3, 1e-1, 1.]:
#     print("Setting reg_lambda to :" + str(v))
#     reg_lambda = v
#     tf_non_linear_model = TFNonLinearModel()
#     train_tf_model(tf_non_linear_model, 10000, 1000)
#     wrapper = TFModelWrapper(tf_non_linear_model)
#     plot_decision_boundary(X, wrapper)