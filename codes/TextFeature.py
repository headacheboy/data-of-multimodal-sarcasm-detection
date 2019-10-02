import tensorflow as tf
import numpy as np

class TextFeature():
    def __init__(self, nHidden, seqLen, guidence, newNet):
        self.nHidden = nHidden
        self.seqLen = seqLen
        tmp = self.getEmbedding()
        self.embedding = tf.Variable(tmp)
        with tf.variable_scope("training_variable"):
            self.weights = {
                "ATT": tf.Variable(tf.truncated_normal(shape=[2*self.nHidden, self.nHidden], stddev=0.08, name="text_att")), 
                "ATTG": tf.Variable(tf.truncated_normal(shape=[200, self.nHidden], stddev=0.08, name="text_att2")),
                "ATTS": tf.Variable(tf.truncated_normal(shape=[self.nHidden, 1], stddev=0.08, name="text_att3")),
                "Fw1": tf.Variable(tf.truncated_normal(shape=[200, self.nHidden], stddev=0.08, name="init_fw1")),
                "Fw2": tf.Variable(tf.truncated_normal(shape=[200, self.nHidden], stddev=0.08, name="init_fw2")),
                "Bw1": tf.Variable(tf.truncated_normal(shape=[200, self.nHidden], stddev=0.08, name="init_bw1")),
                "Bw2": tf.Variable(tf.truncated_normal(shape=[200, self.nHidden], stddev=0.08, name="init_bw2")),
            }
            self.biases = {
                "Fw1": tf.Variable(tf.constant(0.01, shape=[self.nHidden], name="init_Fw1")), 
                "Fw2": tf.Variable(tf.constant(0.01, shape=[self.nHidden], name="init_Fw2")), 
                "Bw1": tf.Variable(tf.constant(0.01, shape=[self.nHidden], name="init_Bw1")), 
                "Bw2": tf.Variable(tf.constant(0.01, shape=[self.nHidden], name="init_Bw2")), 
            }
        self.X = tf.placeholder(tf.int32, [None, self.seqLen])
        self.pKeep = tf.placeholder(tf.float32)
        self.build(guidence, newNet)

    def build(self, guidence, newNet):
        with tf.variable_scope("training_variable"):
            inputEmb = tf.nn.embedding_lookup(self.embedding, self.X)
            initFw = tf.nn.rnn_cell.LSTMStateTuple(tf.nn.relu(tf.matmul(guidence, self.weights["Fw1"]) + self.biases["Fw1"]), tf.nn.relu(tf.matmul(guidence, self.weights["Fw2"]) + self.biases["Fw2"]))
            initBw = tf.nn.rnn_cell.LSTMStateTuple(tf.nn.relu(tf.matmul(guidence, self.weights["Bw1"]) + self.biases["Bw1"]), tf.nn.relu(tf.matmul(guidence, self.weights["Bw2"]) + self.biases["Bw2"]))
            rnnCellFw = tf.contrib.rnn.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(self.nHidden), input_keep_prob=self.pKeep,
                                                      output_keep_prob=1.0)
            rnnCellBw = tf.contrib.rnn.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(self.nHidden), input_keep_prob=self.pKeep,
                                                      output_keep_prob=1.0)
            outputs, state = tf.nn.bidirectional_dynamic_rnn(cell_fw=rnnCellFw, cell_bw=rnnCellBw, inputs=inputEmb, initial_state_fw=initFw, initial_state_bw=initBw,
                                                             dtype=tf.float32)
            outputsConcat = tf.concat(outputs, axis=2)
            self.outputs = outputsConcat
            self.RNNState = tf.reduce_mean(outputsConcat, axis=1)

    def getEmbedding(self):
        return np.loadtxt("../words/vector", delimiter=' ', dtype='float32')
