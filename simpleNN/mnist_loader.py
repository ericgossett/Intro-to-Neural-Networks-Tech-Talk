import pickle
import gzip
import os

import numpy as np

DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'data'
)

def load_data():
    with gzip.open(DATA_PATH + '/mnist.pkl.gz', 'rb') as f:

        training, validation, test = pickle.load(f, encoding='latin1')

        """
        training - an array of tuples with the first entry being a row vector
                   of each 784 pixel value and the second its classification
                   as a single alue (0-9). size=50,000 images

        validation - same format as training, used to tune hyper parameters.
                     size=10,000 images
        test - same format, used to test the final tuned model and compare to
               other tuned models. size=10,000 images
        """

        training_images = [np.reshape(img, (784, 1)) for img in training[0]]
        training_results = []
        for y in training[1]:
            vec = np.zeros((10, 1))
            vec[y] = 1.0
            training_results.append(vec)
        training = list(zip(training_images, training_results))

        validation = list(zip(
            [np.reshape(img, (784, 1)) for img in validation[0]],
            validation[1]
        ))

        test = list(zip(
            [np.reshape(img, (784, 1)) for img in test[0]],
            test[1]
        ))

        return {
            'train': training,
            'validate': validation,
            'test': test
        }
