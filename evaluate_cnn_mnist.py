import theano
import theano.tensor as T
import numpy as np
from theano.tensor.nnet import conv2d, relu
from theano.tensor.signal.pool import pool_2d
import pickle
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

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
valid_set_x, valid_set_y = shared_dataset(valid_set)
test_set_x, test_set_y = shared_dataset(test_set)

def test_noise_robustness():
    test_x = test_set_x.get_value(borrow=True)
    test_y = test_set_y.eval()
    noise = np.random.normal(0, 0.1, test_x.shape)
    noisy_test_x = np.clip(test_x + noise, 0, 1)
    
    noisy_shared_x = theano.shared(noisy_test_x, borrow=True)
    noisy_test_model = theano.function(
        inputs=[index],
        outputs=T.mean(T.eq(predictions, y)),
        givens={
            X: noisy_shared_x[index * batch_size:(index + 1) * batch_size].reshape((batch_size, 1, 28, 28)),
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )
    
    noisy_acc = np.mean([noisy_test_model(i) for i in range(n_test_batches)])
    print(f"Noisy Test Accuracy: {noisy_acc:.4f}")


    
# Define CNN model
def build_cnn(learning_rate=0.01, n_epochs=10, batch_size=32):
    rng = np.random.RandomState(1234)
    
    # Symbolic inputs
    X = T.tensor4('X', dtype=theano.config.floatX)
    y = T.ivector('y')
    
    # Layer 1: Conv + ReLU + MaxPooling
    W1 = theano.shared(np.asarray(rng.uniform(low=-0.1, high=0.1, size=(32, 1, 5, 5)), dtype=theano.config.floatX), name='W1')
    b1 = theano.shared(np.zeros((32,), dtype=theano.config.floatX), name='b1')
    conv1 = conv2d(X, W1, border_mode='valid', input_shape=(batch_size, 1, 28, 28))
    conv1_out = relu(conv1 + b1.dimshuffle('x', 0, 'x', 'x'))
    pool1 = pool_2d(conv1_out, ws=(2, 2), ignore_border=True)
    
    # Layer 2: Conv + ReLU + MaxPooling
    W2 = theano.shared(np.asarray(rng.uniform(low=-0.1, high=0.1, size=(64, 32, 5, 5)), dtype=theano.config.floatX), name='W2')
    b2 = theano.shared(np.zeros((64,), dtype=theano.config.floatX), name='b2')
    conv2 = conv2d(pool1, W2, border_mode='valid')
    conv2_out = relu(conv2 + b2.dimshuffle('x', 0, 'x', 'x'))
    pool2 = pool_2d(conv2_out, ws=(2, 2), ignore_border=True)
    
    # Fully connected layer
    flat_dim = 64 * 4 * 4
    pool2_flat = pool2.reshape((batch_size, flat_dim))
    W3 = theano.shared(np.asarray(rng.uniform(low=-0.1, high=0.1, size=(flat_dim, 10)), dtype=theano.config.floatX), name='W3')
    b3 = theano.shared(np.zeros((10,), dtype=theano.config.floatX), name='b3')
    output = T.nnet.softmax(T.dot(pool2_flat, W3) + b3)
    
    # Loss and predictions
    cost = T.mean(T.nnet.categorical_crossentropy(output, y))
    predictions = T.argmax(output, axis=1)
    params = [W1, b1, W2, b2, W3, b3]
    updates = [(param, param - learning_rate * T.grad(cost, param)) for param in params]
    
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
        outputs=[T.mean(T.eq(predictions, y)), predictions],
        givens={
            X: test_set_x[index * batch_size:(index + 1) * batch_size].reshape((batch_size, 1, 28, 28)),
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )
    
    valid_model = theano.function(
        inputs=[index],
        outputs=[cost, T.mean(T.eq(predictions, y))],
        givens={
            X: valid_set_x[index * batch_size:(index + 1) * batch_size].reshape((batch_size, 1, 28, 28)),
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )
    
    # Training loop
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    
    valid_losses = []
    valid_accuracies = []
    
    for epoch in range(n_epochs):
        for minibatch_index in range(n_train_batches):
            train_model(minibatch_index)
        
        valid_results = [valid_model(i) for i in range(n_valid_batches)]
        valid_loss = np.mean([r[0] for r in valid_results])
        valid_acc = np.mean([r[1] for r in valid_results])
        valid_losses.append(valid_loss)
        valid_accuracies.append(valid_acc)
        
        test_results = [test_model(i) for i in range(n_test_batches)]
        test_acc = np.mean([r[0] for r in test_results])
        test_preds = np.concatenate([r[1] for r in test_results])
        test_true = test_set_y.eval()[:len(test_preds)]
        
        print(f'Epoch {epoch + 1}, Validation Loss: {valid_loss:.4f}, Validation Accuracy: {valid_acc:.4f}, Test Accuracy: {test_acc:.4f}')
    
    # Metrics and plots
    cm = confusion_matrix(test_true, test_preds)
    print("\nClassification Report:")
    print(classification_report(test_true, test_preds, digits=4))
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, n_epochs + 1), valid_losses, label='Validation Loss')
    plt.title('Validation Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, n_epochs + 1), valid_accuracies, label='Validation Accuracy')
    plt.title('Validation Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.close()

if __name__ == '__main__':
    build_cnn()
    visualize_predictions()
    test_noise_robustness()