import tensorflow as tf
import numpy as np

class ExtractFeature():
    def __init__(self):
        self.embedding = self.getEmb()
        self.embSize = self.embedding.shape[1]
        self.vocabSize = self.embedding.shape[0]
        self.x = tf.placeholder(tf.int32, [None, 5])
        with tf.variable_scope("training_variable"):
            self.weights = {
                "MLP1": tf.Variable(tf.truncated_normal(shape=[self.embSize, self.embSize/2], stddev=0.08)), 
                "MLP2": tf.Variable(tf.truncated_normal(shape=[self.embSize/2, 1], stddev=0.08))
            }
            self.biases = {
                "MLP1": tf.Variable(tf.constant(0.01, shape=[self.embSize/2], dtype=tf.float32)), 
                "MLP2": tf.Variable(tf.constant(0.01, shape=[1], dtype=tf.float32))
            }
        self.inputEmb = tf.nn.embedding_lookup(self.embedding, self.x)
        p1 = tf.matmul(tf.reshape(self.inputEmb, [-1, self.embSize]), self.weights["MLP1"])+self.biases["MLP1"]
        p1 = tf.matmul(tf.nn.relu(p1), self.weights["MLP2"])+self.biases["MLP2"]
        p1 = tf.reshape(p1, [-1, 5])
        p1 = tf.reshape(tf.nn.softmax(p1), [-1, 1, 5])
        self.finalState = tf.reshape(tf.matmul(p1, self.inputEmb), [-1, self.embSize])

    def getEmb(self):
        return np.loadtxt("../ExtractWords/vector", delimiter=' ', dtype='float32')
