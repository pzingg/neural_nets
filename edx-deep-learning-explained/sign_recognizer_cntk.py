
# coding: utf-8

# In[1]:


# # Lab 3 - Multi Layer Perceptron with SIGNS
#
# This lab corresponds to Module 3 of the "Deep Learning Explained" course.  We assume
# that you have successfully completed Lab 1 (Downloading the SIGNS data).
#
# In this lab, we train a multi-layer perceptron on SIGNS data. This notebook provides
# the recipe using Python APIs.


# **Goal**:
# Our goal is to train a classifier that will identify the digits in the SIGNS dataset. Additionally, we aspire to achieve lower error rate with Multi-layer perceptron compared to Multi-class logistic regression.
#
# **Approach**:
# There are 4 stages in this lab:
# - **Data reading**: We will use the CNTK Text reader.
# - **Data preprocessing**: Covered in part A (suggested extension section).
# - **Model creation**: Multi-Layer Perceptron model.
# - **Train-Test-Predict**: This is the same workflow introduced in the lectures
#

# In[3]:


import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

import cntk as C


# In the block below, we check if we are running this notebook in the CNTK internal test
# machines by looking for environment variables defined there. We then select the right
# target device (GPU vs CPU) to test this notebook. In other cases, we use CNTK's default
# policy to use the best available device (GPU, if available, else CPU).

# In[4]:


# Select the right target device when this notebook is being tested:
if 'TEST_DEVICE' in os.environ:
    if os.environ['TEST_DEVICE'] == 'cpu':
        C.device.try_set_default_device(C.device.cpu())
    else:
        C.device.try_set_default_device(C.device.gpu(0))


# In[6]:


# Ensure we always get the same amount of randomness
np.random.seed(0)
C.cntk_py.set_fixed_random_seed(1)
C.cntk_py.force_deterministic_algorithms()


# In[8]:


# Ensure the training and test data is generated and available for this tutorial.
# We search in two locations in the toolkit for the cached SIGN data set.
data_found = False
for data_dir in ["../datasets"]:
    train_file = os.path.join(data_dir, "Train-hands-64x64x3-cntk.txt")
    test_file = os.path.join(data_dir, "Test-hands-64x64x3-cntk.txt")
    if os.path.isfile(train_file) and os.path.isfile(test_file):
        data_found = True
        break
if not data_found:
    raise ValueError("Please generate the data by running data_loader.py")
print("Data directory is {0}".format(data_dir))



# ## Data reading
#
# There are different ways one can read data into CNTK. The easiest way is to load the
# data in memory using NumPy / SciPy / Pandas readers. However, this can be done only
# for small data sets. Since deep learning requires large amount of data we have chosen
# in this course to show how to leverage built-in distributed readers that can scale to
# terabytes of data with little extra effort.
#
# We are using the SIGNS data from the Coursera tutorial.
# The dataset has 1080 training images and 120 test images with each image being
# 64 x 64 pixels, and each pixel represented as 3 RGB values between 0 and 1.
# Thus the number of features is equal to 12228 (= 64 x 64 x 3),
# 1 per pixel. The variable `num_output_classes` is set to 6 corresponding to the
# number of signed "digitst" (0 - 5) in the dataset.
#
# In Lab 1, the data was downloaded and written to 2 CTF (CNTK Text Format) files,
#  1 for training, and 1 for testing. Each line of these text files takes the form:
#
#     |labels 0 0 0 1 0 0 0 0 0 0 |features 0.5 0.5 0.5 0.5 ...
#                                                   (12228 floats each representing a RGB value)
#
# We are going to use the image pixels corresponding the integer stream named "features".
# We define a `create_reader` function to read the training and test data using the
# [CTF deserializer](https://cntk.ai/pythondocs/cntk.io.html?highlight=ctfdeserializer#cntk.io.CTFDeserializer).
# The labels are [1-hot encoded](https://en.wikipedia.org/wiki/One-hot).
# Refer to Lab 1 for data format visualizations.

# In[7]:


# Read a CTF formatted text (as mentioned above) using the CTF deserializer from a file
def create_reader(path, is_training, input_dim, num_label_classes):
    return C.io.MinibatchSource(C.io.CTFDeserializer(path, C.io.StreamDefs(
        labels = C.io.StreamDef(field='labels', shape=num_label_classes, is_sparse=False),
        features   = C.io.StreamDef(field='features', shape=input_dim, is_sparse=False)
    )), randomize = is_training, max_sweeps = C.io.INFINITELY_REPEAT if is_training else 1)



# First let us create some helper functions that will be needed to visualize different
# functions associated with training.

# In[16]:


# Define a utility function to compute the moving average sum.
# A more efficient implementation is possible with np.cumsum() function
def moving_average(a, w=5):
    if len(a) < w:
        return a[:]    # Need to send a copy of the array
    return [val if idx < w else sum(a[(idx-w):idx])/w for idx, val in enumerate(a)]


# Defines a utility that prints the training progress
def print_training_progress(trainer, mb, frequency, verbose=1):
    training_loss = "NA"
    eval_error = "NA"

    if mb%frequency == 0:
        training_loss = trainer.previous_minibatch_loss_average
        eval_error = trainer.previous_minibatch_evaluation_average
        if verbose:
            print("Minibatch: {0}, Loss: {1:.4f}, Error: {2:.2f}%".format(mb, training_loss, eval_error*100))

    return mb, training_loss, eval_error



# ## Network Creation
#
# Our multi-layer perceptron will be relatively simple with 2 hidden layers
# (`num_hidden_layers`). The number of nodes in the hidden layer being a parameter
# specified by `hidden_layers_dim`. The figure below illustrates the entire model we will use
# in this tutorial in the context of SIGN data.

# If you are not familiar with the terms *hidden_layer* and *number of hidden layers*,
# please review the module 3 course videos.
#
# Each Dense layer (as illustrated below) shows the input dimensions, output dimensions
# and activation function that layer uses. Specifically, the layer below shows: input
# dimension = 12228 (1 dimension for each RGB pixel value), output dimension = 25 or 12
# (number of hidden nodes, a parameter specified by the user) and activation function being
# [relu](https://cntk.ai/pythondocs/cntk.ops.html?highlight=relu#cntk.ops.relu).
#
# In this model we have 2 dense layer called the hidden layers each with an activation
# function of `relu`.  These are followed by the dense output layer with no activation.
#
# The output dimensions (a.k.a. number of hidden nodes) in the 2 hidden layers are set to 25
# for the first hidden layer and 12 for the second hidden layer. The number of hidden layers is 2.
#
# The final output layer emits a vector of 6 values. Since we will be using softmax to
# normalize the output of the model we do not use an activation function in this layer.
# The softmax operation comes bundled with the
# [loss function](https://cntk.ai/pythondocs/cntk.losses.html) we will be using later
# in this tutorial.

# In[9]:



def create_network(num_output_classes, features, hidden_layers_dims = [25, 12]):
    num_hidden_layers = len(hidden_layers_dims)
    with C.layers.default_options(init = C.layers.glorot_uniform(), activation = C.ops.relu):
            h = features
            for i in range(num_hidden_layers):
                h = C.layers.Dense(hidden_layers_dims[i])(h)
            r = C.layers.Dense(num_output_classes, activation = None)(h)
            return r



# Make a function that matches up with the Coursera tutorial
# Define the data dimensions and learning rate
def model(train_file, test_file,
    num_train_samples = 1080,
    train_minibatch_size = 32,
    num_test_samples = 120,
    test_minibatch_size = 8,
    input_dim = 64*64*3,
    num_output_classes = 6,
    learning_rate = 0.0001,
    num_epochs = 1500,
    print_cost = True):

    # Network input and output:
    # - **input** variable (a key CNTK concept):
    # >An **input** variable is a container in which we fill different observations in this case
    # image pixels during model learning (a.k.a.training) and model evaluation (a.k.a. testing).
    # Thus, the shape of the `input` must match the shape of the data that will be provided.
    # For example, when data are images each of  height 10 pixels and width 5 pixels, the input
    # feature dimension will be 50 (representing the total number of image pixels). More on data
    # and their dimensions to appear in separate tutorials.
    #
    #
    # **Knowledge Check** What is the input dimension of your chosen model? This is fundamental
    # to our understanding of variables in a network or model representation in CNTK.
    #

    # In[10]:


    input = C.input_variable(input_dim)
    label = C.input_variable(num_output_classes)


    # ## Multi-layer Perceptron setup
    #
    # The code below is a direct translation of the model shown above.

    # In[11]:


    # `z` will be used to represent the output of a network.
    #
    # We introduced sigmoid function in CNTK 102, in this tutorial you should try
    # different activation functions in the hidden layer. You may choose to do this
    # right away and take a peek into the performance later in the tutorial or run
    # the preset tutorial and then choose to perform the suggested exploration.
    #
    #
    # ** Suggested Exploration **
    # - Record the training error you get with `sigmoid` as the activation function
    # - Now change to `relu` as the activation function and see if you can improve
    # your training error
    #
    # **Knowledge Check**: Name some of the different supported activation functions.
    # Which activation function gives the least training error?

    # In[12]:


    # Scale the input to 0-1 range by dividing each pixel by 255.
    # If using 8-bit grayscale images, such as MNIST, we need to scale the feature vectors.
    # z = create_network(num_output_classes, input/255.0)

    # Since SIGNS dataset is pre-scaled (RGB 0-1), we don't scale these.
    z = create_network(num_output_classes, input)


    # ## Training

    # Below, we define the **Loss** function, which is used to guide weight changes
    # during training.
    #
    # As explained in the lectures, we use the `softmax` function to map the accumulated
    # evidences or activations to a probability distribution over the classes
    # (Details of the [softmax function][] and other [activation][] functions).
    #
    # [softmax function]: http://cntk.ai/pythondocs/cntk.ops.html#cntk.ops.softmax
    # [activation]: https://github.com/Microsoft/CNTK/wiki/Activation-Functions
    # We minimize the cross-entropy between the label and predicted probability by the network.
    #

    # In[13]:


    loss = C.cross_entropy_with_softmax(z, label)


    # #### Evaluation
    #
    # Below, we define the **Evaluation** (or metric) function that is used to report a
    # measurement of how well our model is performing.
    #
    # For this problem, we choose the **classification_error()** function as our metric,
    # which returns the average error over the associated samples (treating a match as "1",
    # where the model's prediction matches the "ground truth" label, and a non-match as "0").

    # In[14]:


    label_error = C.classification_error(z, label)


    # ### Configure training
    #
    # The trainer strives to reduce the `loss` function by different optimization approaches,
    # [Stochastic Gradient Descent][] (`sgd`) being a basic one. Typically, one would start
    # with random initialization of the model parameters. The `sgd` optimizer would calculate
    # the `loss` or error between the predicted label against the corresponding ground-truth
    # label and using [gradient-decent][] generate a new set model parameters in a single iteration.
    #
    # The aforementioned model parameter update using a single observation at a time is
    # attractive since it does not require the entire data set (all observation) to be loaded
    # in memory and also requires gradient computation over fewer datapoints, thus allowing
    # for training on large data sets. However, the updates generated using a single observation
    # sample at a time can vary wildly between iterations. An intermediate ground is to load
    # a small set of observations and use an average of the `loss` or error from that set to
    # update the model parameters. This subset is called a *minibatch*.
    #
    # With minibatches we often sample observation from the larger training dataset.
    # We repeat the process of model parameters update using different combination of
    # training samples and over a period of time minimize the `loss` (and the error).
    # When the incremental error rates are no longer changing significantly or after a
    # preset number of maximum minibatches to train, we claim that our model is trained.
    #
    # One of the key parameter for optimization is called the `learning_rate`.
    # For now, we can think of it as a scaling factor that modulates how much we change the
    # parameters in any iteration. We will be covering more details in later tutorial.
    # With this information, we are ready to create our trainer.
    #
    # [optimization]: https://en.wikipedia.org/wiki/Category:Convex_optimization
    # [Stochastic Gradient Descent]: https://en.wikipedia.org/wiki/Stochastic_gradient_descent
    # [gradient-decent]: http://www.statisticsviews.com/details/feature/5722691/Getting-to-the-Bottom-of-Regression-with-Gradient-Descent.html

    # In[15]:


    # Instantiate the trainer object to drive the model training
    lr_schedule = C.learning_rate_schedule(learning_rate, C.UnitType.minibatch)
    learner = C.sgd(z.parameters, lr_schedule)
    trainer = C.Trainer(z, (loss, label_error), [learner])


    # ### Run the trainer
    #
    # We are now ready to train our fully connected neural net. We want to decide what
    # data we need to feed into the training engine.
    #
    # In this example, each iteration of the optimizer will work on `train_minibatch_size`
    # sized samples. We would like to train on all 60000 observations. Additionally we will
    # make multiple passes through the data specified by the variable `num_epochs`.
    # With these parameters we can proceed with training our simple multi-layer perceptron network.

    # In[17]:


    # Initialize the parameters for the trainer

    num_minibatches_to_train = (num_train_samples * num_epochs) // train_minibatch_size
    print("Will use {} minibatches for training".format(num_minibatches_to_train))


    # In[18]:


    # Create the reader to training data set
    reader_train = create_reader(train_file, True, input_dim, num_output_classes)

    # Map the data streams to the input and labels.
    input_map = {
        label  : reader_train.streams.labels,
        input  : reader_train.streams.features
    }

    # Run the trainer on and perform model training
    training_progress_output_freq = 2000

    plotdata = {"batchsize":[], "loss":[], "error":[]}

    for i in range(0, int(num_minibatches_to_train)):

        # Read a mini batch from the training data file
        data = reader_train.next_minibatch(train_minibatch_size, input_map = input_map)

        trainer.train_minibatch(data)
        batchsize, loss, error = print_training_progress(trainer, i, training_progress_output_freq, verbose=1)

        if not (loss == "NA" or error == "NA"):
            plotdata["batchsize"].append(batchsize)
            plotdata["loss"].append(loss)
            plotdata["error"].append(error)


    if print_cost:
        # Let us plot the errors over the different training minibatches. Note that as we
        # iterate the training loss decreases though we do see some intermediate bumps.

        # In[19]:

        # Compute the moving average loss to smooth out the noise in SGD
        plotdata["avgloss"] = moving_average(plotdata["loss"])
        plotdata["avgerror"] = moving_average(plotdata["error"])

        plt.figure(1)
        plt.subplot(211)
        plt.plot(plotdata["batchsize"], plotdata["avgloss"], 'b--')
        plt.xlabel('Minibatch number')
        plt.ylabel('Loss')
        plt.title('Minibatch run vs. Training loss')
        plt.show()

        plt.subplot(212)
        plt.plot(plotdata["batchsize"], plotdata["avgerror"], 'r--')
        plt.xlabel('Minibatch number')
        plt.ylabel('Label Prediction Error')
        plt.title('Minibatch run vs. Label Prediction Error')
        plt.show()


    final_loss = trainer.previous_minibatch_loss_average
    avg_training_error = trainer.previous_minibatch_evaluation_average


    # ## Evaluation / Testing
    #
    # Now that we have trained the network, let us evaluate the trained network on the
    # test data. This is done using `trainer.test_minibatch`.

    # In[20]:


    # Read the test data
    reader_test = create_reader(test_file, False, input_dim, num_output_classes)

    test_input_map = {
        label  : reader_test.streams.labels,
        input  : reader_test.streams.features,
    }

    # Test data for trained model
    num_minibatches_to_test = num_test_samples // test_minibatch_size
    avg_test_error = 0.0

    for i in range(num_minibatches_to_test):

        # We are loading test data in batches specified by test_minibatch_size
        # Each data point in the minibatch is a SIGNS digit image of 12228 dimensions
        # with one RGB value per dimension that we will encode / decode with the
        # trained model.
        data = reader_test.next_minibatch(test_minibatch_size,
                                          input_map = test_input_map)

        eval_error = trainer.test_minibatch(data)
        avg_test_error = avg_test_error + eval_error

    avg_test_error = avg_test_error / num_minibatches_to_test

    # Average of evaluation errors of all test minibatches
    print("Training Loss: {0:.2f}%".format(final_loss))
    print("Training Accuracy: {0:.2f}%".format(100.0 * (1.0 - avg_training_error)))
    print("Test Accuracy: {0:.2f}%".format(100.0 * (1.0 - avg_test_error)))
    print("Average Test Error: {0:.2f}%".format(100.0 * avg_test_error))


    # Note, this error is very comparable to our training error indicating that our
    # model has good "out of sample" error a.k.a. generalization error. This implies
    # that our model can very effectively deal with previously unseen observations
    # (during the training process). This is key to avoid the phenomenon of overfitting.
    #
    # This is a **huge** reduction in error compared to multi-class LR (from Lab 02).
    return z


# We have so far been dealing with aggregate measures of error. Let us now get the
# probabilities associated with individual data points. For each observation, the
# `eval` function returns the probability distribution across all the classes.
# The classifier is trained to recognize digits, hence has 6 classes. First let us
# route the network output through a `softmax` function. This maps the aggregated
# activations across the network to probabilities across the 6 classes.

# In[21]:

# Let us test a small minibatch sample from the test data.
def batch_evaluate(z, test_file,
    input_dim = 64*64*3,
    num_output_classes = 6,
    eval_minibatch_size = 25):

    # In[22]:

    input = C.input_variable(input_dim)
    label = C.input_variable(num_output_classes)

    # Read the training data
    reader_test = create_reader(test_file, False, input_dim, num_output_classes)

    test_input_map = {
        label : reader_test.streams.labels,
        input : reader_test.streams.features,
    }

    data = reader_test.next_minibatch(eval_minibatch_size, input_map = test_input_map)

    img_label = data[label].asarray()
    img_data = data[input].asarray()

    out = C.softmax(z)
    predicted_label_prob = [out.eval(img_data[i]) for i in range(len(img_data))]

    # Find the index with the maximum value for both predicted as well as the ground truth
    pred = [np.argmax(predicted_label_prob[i]) for i in range(len(predicted_label_prob))]
    gtlabel = [np.argmax(img_label[i]) for i in range(len(img_label))]

    print("   Batch Labels: ", gtlabel[:eval_minibatch_size])
    print("Batch Predicted: ", pred)


# Visualize one of the test images and make a prediction.
def predict(z, test_file, sample_number,
    input_dim = 64*64*3,
    num_output_classes = 6):

    # Read the data for evaluation
    reader_eval = create_reader(test_file, False, input_dim, num_output_classes)

    eval_input_map = {
        input : reader_eval.streams.features
    }

    # Skip the first n images
    if sample_number > 0:
        _ = reader_eval.next_minibatch(sample_number, input_map = eval_input_map)

    # Now we can read the next image into an array of length 1
    data = reader_eval.next_minibatch(1, input_map = eval_input_map)

    img_data = data[input].asarray()

    # Plot the image
    plt.imshow(img_data[0].reshape(64, 64, 3))
    plt.axis('off')
    plt.title('Test Image No. {}'.format(sample_number))
    plt.show()

    out = C.softmax(z)
    predicted_label_prob = out.eval(img_data[0])

    # Find the highest probability index
    img_pred = np.argmax(predicted_label_prob)

    print("Image {}, p: {}, Predicted: {}".format(sample_number, predicted_label_prob, img_pred))
    return img_pred


if __name__ == "__main__":
    z_out = model(train_file, test_file, num_epochs = 200, print_cost = False)
    # batch_evaluate(z_out, test_file)
    predict(z_out, test_file, np.random.randint(120))


# **Suggested Explorations**
# -  Try exploring how the classifier behaves with different parameters - suggest
# changing the `minibatch_size` parameter from 25 to say 64 or 128. What happens to the
# error rate? How does the error compare to the logistic regression classifier?
# - Try increasing the number of sweeps
# - Can you change the network to reduce the training error rate? When do you see
#  *overfitting* happening?
