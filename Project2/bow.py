from utils import load_cifar10_data
from utils import extract_sift_descriptors
from utils import build_codebook
from utils import input_vector_encoder
from classifier import svm_classifier

import numpy as np


VOC_SIZE = 100



if __name__ == '__main__':

    # Training
    x_train, y_train = load_cifar10_data(dataset='train')
    x_test, y_test = load_cifar10_data(dataset='test')

    print "SIFT feature extraction"
    x_train = [extract_sift_descriptors(img) for img in x_train]
    x_test = [extract_sift_descriptors(img) for img in x_test]

    # Remove None in SIFT extraction
    x_train = [each for each in zip(x_train, y_train) if each[0] != None]
    x_train, y_train = zip(*x_train)
    x_test = [each for each in zip(x_test, y_test) if each[0] != None]
    x_test, y_test = zip(*x_test)

    print "Train/Test split: {:d}/{:d}".format(len(y_train), len(y_test))
    print "Codebook Size: {:d}".format(VOC_SIZE)

    print "Building the codebook, it will take some time"
    codebook = build_codebook(x_train, voc_size=VOC_SIZE)
    import cPickle
    with open('./bow_codebook.pkl','w') as f:
        cPickle.dump(codebook, f)

    print "Bag of words encoding"
    x_train = [input_vector_encoder(x, codebook) for x in x_train]
    x_train = np.asarray(x_train)

    x_test = [input_vector_encoder(each, codebook) for each in x_test]
    x_test = np.asarray(x_test)

    svm_classifier(x_train, y_train, x_test, y_test)
