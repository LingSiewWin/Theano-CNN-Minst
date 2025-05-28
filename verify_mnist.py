import pickle
import numpy

with open('/Users/lingsiewwin/Documents/Github/Theano_CNN_Mist/mnist.pkl', 'rb') as f:
    data = pickle.load(f, encoding='latin1')
print(len(data), [len(d) for d in data], [d[0].shape for d in data])