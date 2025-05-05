from urllib import request
import numpy as np
import gzip
import pickle
import os

# from 'https://github.com/hsjeong5/MNIST-for-Numpy/blob/master/mnist.py'

names = [("https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz", "train_images.gz", "train_images"),
         ("https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz", "test_images.gz", "test_images"),
         ("https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz", "train_labels.gz", "train_labels"),
         ("https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz", "test_labels.gz", "test_labels")]


# download files
def download():
    newpath = os.path.abspath('mnist')
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    for name in names:
        print("downloading from", name[0])
        request.urlretrieve(name[0], newpath + "/" + name[1])


# save images
def save():
    path = os.path.abspath('mnist')
    mnist = {}
    for name in names[:2]: 
        with gzip.open(path + '/' + name[1], 'rb') as f:
            mnist[name[2]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28*28)
    for name in names[-2:]:
        with gzip.open(path + '/' + name[1], 'rb') as f:
            mnist[name[2]] = np.frombuffer(f.read(), np.uint8, offset=8)
    with open(path + '/' + "mnist.pkl", 'wb') as f:
        pickle.dump(mnist, f)
    print("dataset saved.")


def load():
    path = os.path.abspath('mnist')
    with open(path + '/' + "mnist.pkl", 'rb') as f:
        mnist = pickle.load(f)
    return mnist["train_images"], mnist["test_images"], mnist["train_labels"], mnist["test_labels"]


def run():
    download()
    save()


if __name__ == '__main__':
    run()


