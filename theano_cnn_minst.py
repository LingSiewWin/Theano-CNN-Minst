import theano
import theano.tensor as T
import numpy as np
from theano.tensor.nnet import conv2d, relu
from theano.tensor.signal.pool import pool_2d
import pickle

# Load MNIST dataset
def load_mnist():
    with open('/Users/lingsiewwin/Documents/Github/Theano_CNN_Mist/mnist.pkl', 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    return train_set, valid_set, test_set

# Shared variable helper
def shared_dataset(data_xy):
    data_x, data_y = data_xy
    assert data_x.shape[1] == 784, f"Expected 784 features, got {data_x.shape[1]}"
    shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX))
    shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX))
    return shared_x, T.cast(shared_y, 'int32')

# Load and prepare data
train_set, valid_set, test_set = load_mnist()
train_set_x, train_set_y = shared_dataset(train_set)
test_set_x, test_set_y = shared_dataset(test_set)

# Define CNN model
def build_cnn(learning_rate=0.01, n_epochs=10, batch_size=32):
    rng = np.random.RandomState(1234)
    
    # Symbolic inputs
    X = T.tensor4('X', dtype=theano.config.floatX)  # (batch_size, 1, 28, 28)
    y = T.ivector('y')  # Labels
    
    # Layer 1: Conv + ReLU + MaxPooling
    W1 = theano.shared(
        np.asarray(rng.uniform(low=-0.1, high=0.1, size=(32, 1, 5, 5)), dtype=theano.config.floatX),
        name='W1'
    )
    b1 = theano.shared(np.zeros((32,), dtype=theano.config.floatX), name='b1')
    conv1 = conv2d(X, W1, border_mode='valid', input_shape=(batch_size, 1, 28, 28))
    conv1_out = relu(conv1 + b1.dimshuffle('x', 0, 'x', 'x'))
    pool1 = pool_2d(conv1_out, ws=(2, 2), ignore_border=True)
    
    # Layer 2: Conv + ReLU + MaxPooling
    W2 = theano.shared(
        np.asarray(rng.uniform(low=-0.1, high=0.1, size=(64, 32, 5, 5)), dtype=theano.config.floatX),
        name='W2'
    )
    b2 = theano.shared(np.zeros((64,), dtype=theano.config.floatX), name='b2')
    conv2 = conv2d(pool1, W2, border_mode='valid')
    conv2_out = relu(conv2 + b2.dimshuffle('x', 0, 'x', 'x'))
    pool2 = pool_2d(conv2_out, ws=(2, 2), ignore_border=True)
    
    # Compute flattened dimension
    flat_dim = 64 * 4 * 4  # (28-5+1)/2=12, (12-5+1)/2=4
    pool2_flat = pool2.reshape((batch_size, flat_dim))
    
    # Fully connected layer
    W3 = theano.shared(
        np.asarray(rng.uniform(low=-0.1, high=0.1, size=(flat_dim, 10)), dtype=theano.config.floatX),
        name='W3'
    )
    b3 = theano.shared(np.zeros((10,), dtype=theano.config.floatX), name='b3')
    output = T.nnet.softmax(T.dot(pool2_flat, W3) + b3)
    
    # Loss and gradients
    cost = T.mean(T.nnet.categorical_crossentropy(output, y))
    params = [W1, b1, W2, b2, W3, b3]
    updates = [
        (param, param - learning_rate * T.grad(cost, param))
        for param in params
    ]
    
    # Compile functions
    index = T.lscalar()
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            X: train_set_x[index * batch_size:(index + 1) * batch_size].reshape((batch_size, 1, 28, 28)),
            y: train_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )
    
    test_model = theano.function(
        inputs=[index],
        outputs=T.mean(T.eq(T.argmax(output, axis=1), y)),
        givens={
            X: test_set_x[index * batch_size:(index + 1) * batch_size].reshape((batch_size, 1, 28, 28)),
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )
    
    # Training loop
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size
    
    for epoch in range(n_epochs):
        for minibatch_index in range(n_train_batches):
            cost = train_model(minibatch_index)
        test_accuracy = np.mean([test_model(i) for i in range(n_test_batches)])
        print(f'Epoch {epoch + 1}, Test Accuracy: {test_accuracy:.4f}')

# Run the model
if __name__ == '__main__':
    build_cnn()