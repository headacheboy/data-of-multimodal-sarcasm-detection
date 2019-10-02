import tensorflow as tf
import ImageFeature
import TextFeature
import ExtractFeature

class Net():
    def __init__(self, nHidden, seqLen):
        self.representation_score = {}
        self.y = tf.placeholder(tf.float32, shape=[None, 1])
        self.extractFeature = ExtractFeature.ExtractFeature()
        self.imageFeature = ImageFeature.ImageFeature()
        newNet = tf.reduce_mean(self.imageFeature.outputLS, axis=0) 
        self.textFeature = TextFeature.TextFeature(nHidden, seqLen, self.extractFeature.finalState, newNet)
        self.l2_para = 1e-7
        with tf.variable_scope("training_variable"):
            
            self.weights = {
                "MLP1": tf.Variable(tf.truncated_normal(shape=[512, 256],
                                                        stddev=0.08, name="MLP1_W")),
                "MLP2": tf.Variable(tf.truncated_normal(shape=[256, 1], 
                                                        stddev=0.08, name="MLP2_W")), 
                "ATT_attr1_1": tf.Variable(tf.truncated_normal(shape=[self.imageFeature.defaultFeatureSize+self.extractFeature.embSize, self.imageFeature.defaultFeatureSize/2+self.extractFeature.embSize/2], stddev=0.08, name="ATT_attr1_1")), 
                "ATT_attr1_2": tf.Variable(tf.truncated_normal(shape=[self.textFeature.nHidden*2+self.extractFeature.embSize, self.textFeature.nHidden+self.extractFeature.embSize/2], stddev=0.08, name="ATT_attr1_2")), 
                "ATT_attr1_3": tf.Variable(tf.truncated_normal(shape=[2*self.extractFeature.embSize, self.extractFeature.embSize], stddev=0.08, name="ATT_attr1_3")), 
                "ATT_attr2_1": tf.Variable(tf.truncated_normal(shape=[ self.imageFeature.defaultFeatureSize/2+self.extractFeature.embSize/2, 1], stddev=0.08, name="ATT_attr2_1")), 
                "ATT_attr2_2": tf.Variable(tf.truncated_normal(shape=[ self.textFeature.nHidden+self.extractFeature.embSize/2, 1], stddev=0.08, name="ATT_attr2_2")), 
                "ATT_attr2_3": tf.Variable(tf.truncated_normal(shape=[ self.extractFeature.embSize, 1], stddev=0.08, name="ATT_attr2_3")), 
                "ATT_img1_1": tf.Variable(tf.truncated_normal(shape=[self.imageFeature.defaultFeatureSize+self.textFeature.nHidden*2, self.imageFeature.defaultFeatureSize/2+self.textFeature.nHidden], stddev=0.08, name="ATT_image1_1")), 
                "ATT_img1_2": tf.Variable(tf.truncated_normal(shape=[self.imageFeature.defaultFeatureSize+self.extractFeature.embSize, self.imageFeature.defaultFeatureSize/2+self.extractFeature.embSize/2], stddev=0.08, name="ATT_image1_2")), 
                "ATT_img1_3": tf.Variable(tf.truncated_normal(shape=[self.imageFeature.defaultFeatureSize*2, self.imageFeature.defaultFeatureSize], stddev=0.08, name="ATT_image1_3")), 
                "ATT_img2_1": tf.Variable(tf.truncated_normal(shape=[ self.imageFeature.defaultFeatureSize/2+self.textFeature.nHidden, 1], stddev=0.08, name="ATT_image2_1")), 
                "ATT_img2_2": tf.Variable(tf.truncated_normal(shape=[ self.imageFeature.defaultFeatureSize/2+self.extractFeature.embSize/2, 1], stddev=0.08, name="ATT_image2_2")), 
                "ATT_img2_3": tf.Variable(tf.truncated_normal(shape=[ self.imageFeature.defaultFeatureSize, 1], stddev=0.08, name="ATT_image2_3")), 
                "ATT_text1_1": tf.Variable(tf.truncated_normal(shape=[self.imageFeature.defaultFeatureSize+self.textFeature.nHidden*2, self.imageFeature.defaultFeatureSize/2+self.textFeature.nHidden], stddev=0.08, name="ATT_text1_1")), 
                "ATT_text1_2": tf.Variable(tf.truncated_normal(shape=[self.textFeature.nHidden*2+self.extractFeature.embSize, self.textFeature.nHidden+self.extractFeature.embSize/2], stddev=0.08, name="ATT_text1_2")), 
                "ATT_text1_3": tf.Variable(tf.truncated_normal(shape=[self.textFeature.nHidden*4, self.textFeature.nHidden*2], stddev=0.08, name="ATT_text1_3")), 
                "ATT_text2_1": tf.Variable(tf.truncated_normal(shape=[ self.imageFeature.defaultFeatureSize/2+self.textFeature.nHidden, 1], stddev=0.08, name="ATT_text2_1")), 
                "ATT_text2_2": tf.Variable(tf.truncated_normal(shape=[ self.textFeature.nHidden+self.extractFeature.embSize/2, 1], stddev=0.08, name="ATT_text2_2")), 
                "ATT_text2_3": tf.Variable(tf.truncated_normal(shape=[ self.textFeature.nHidden*2, 1], stddev=0.08, name="ATT_text2_3")), 
                "ATT_WI1": tf.Variable(tf.truncated_normal(shape=[self.imageFeature.defaultFeatureSize, 512], stddev=0.08, name="ATT_WI")), 
                "ATT_WT1": tf.Variable(tf.truncated_normal(shape=[2*nHidden, 512], stddev=0.08, name="ATT_WT")), 
                "ATT_WA1": tf.Variable(tf.truncated_normal(shape=[200, 512], stddev=0.08, name="ATT_WA")), 
                "ATT_WI2": tf.Variable(tf.truncated_normal(shape=[self.imageFeature.defaultFeatureSize, 512], stddev=0.08, name="ATT_WI2")), 
                "ATT_WT2": tf.Variable(tf.truncated_normal(shape=[2*nHidden, 512], stddev=0.08, name="ATT_WT2")), 
                "ATT_WA2": tf.Variable(tf.truncated_normal(shape=[200, 512], stddev=0.08, name="ATT_WA2")), 
                "ATT_WF_1": tf.Variable(tf.truncated_normal(shape=[512, 1], stddev=0.08, name="ATT_WF_1")), 
                "ATT_WF_2": tf.Variable(tf.truncated_normal(shape=[512, 1], stddev=0.08, name="ATT_WF_2")), 
                "ATT_WF_3": tf.Variable(tf.truncated_normal(shape=[512, 1], stddev=0.08, name="ATT_WF_3")), 
            }
            self.biases = {
                "MLP1": tf.Variable(tf.constant(0.01, shape=[256], dtype=tf.float32, name="MLP1_b")),
                "MLP2": tf.Variable(tf.constant(0.01, shape=[1], dtype=tf.float32, name="MLP2_b")),
                "ATT_attr1_1": tf.Variable(tf.constant(0.01, shape=[self.imageFeature.defaultFeatureSize/2+self.extractFeature.embSize/2], name="ATT_attr1_1")), 
                "ATT_attr1_2": tf.Variable(tf.constant(0.01, shape=[self.textFeature.nHidden+self.extractFeature.embSize/2], name="ATT_attr1_2")), 
                "ATT_attr1_3": tf.Variable(tf.constant(0.01, shape=[self.extractFeature.embSize], name="ATT_attr1_3")), 
                "ATT_attr2_1": tf.Variable(tf.constant(0.01, shape=[1], name="ATT_attr2_1")), 
                "ATT_attr2_2": tf.Variable(tf.constant(0.01, shape=[1], name="ATT_attr2_2")), 
                "ATT_attr2_3": tf.Variable(tf.constant(0.01, shape=[1], name="ATT_attr2_3")), 
                "ATT_img1_1": tf.Variable(tf.constant(0.01, shape=[self.imageFeature.defaultFeatureSize/2+self.textFeature.nHidden], name="ATT_image1_1")), 
                "ATT_img1_2": tf.Variable(tf.constant(0.01, shape=[self.imageFeature.defaultFeatureSize/2+self.extractFeature.embSize/2], name="ATT_image1_2")), 
                "ATT_img1_3": tf.Variable(tf.constant(0.01, shape=[self.imageFeature.defaultFeatureSize], name="ATT_image1_3")), 
                "ATT_img2_1": tf.Variable(tf.constant(0.01, shape=[1], name="ATT_image2_1")), 
                "ATT_img2_2": tf.Variable(tf.constant(0.01, shape=[1], name="ATT_image2_2")), 
                "ATT_img2_3": tf.Variable(tf.constant(0.01, shape=[1], name="ATT_image2_3")), 
                "ATT_text1_1": tf.Variable(tf.constant(0.01, shape=[self.imageFeature.defaultFeatureSize/2+self.textFeature.nHidden], name="ATT_text1_1")), 
                "ATT_text1_2": tf.Variable(tf.constant(0.01, shape=[self.textFeature.nHidden+self.extractFeature.embSize/2], name="ATT_text1_2")), 
                "ATT_text1_3": tf.Variable(tf.constant(0.01, shape=[self.textFeature.nHidden*2], name="ATT_text1_3")), 
                "ATT_text2_1": tf.Variable(tf.constant(0.01, shape=[1], name="ATT_text2_1")), 
                "ATT_text2_2": tf.Variable(tf.constant(0.01, shape=[1], name="ATT_text2_2")), 
                "ATT_text2_3": tf.Variable(tf.constant(0.01, shape=[1], name="ATT_text2_3")), 
                "ATT_WW": tf.Variable(tf.constant(0.01, shape=[512], name="ATT_WW")), 
                "ATT_WI": tf.Variable(tf.constant(0.01, shape=[512], name="ATT_WI")), 
                "ATT_WT": tf.Variable(tf.constant(0.01, shape=[512], name="ATT_WT")), 
                "ATT_WI1": tf.Variable(tf.constant(0.01, shape=[512], name="ATT_WI1")), 
                "ATT_WT1": tf.Variable(tf.constant(0.01, shape=[512], name="ATT_WT1")), 
                "ATT_WA": tf.Variable(tf.constant(0.01, shape=[512], name="ATT_WA")), 
                "ATT_WF_1": tf.Variable(tf.constant(0.01, shape=[1], name="ATT_WF_1")), 
                "ATT_WF_2": tf.Variable(tf.constant(0.01, shape=[1], name="ATT_WF_2")), 
                "ATT_WF_3": tf.Variable(tf.constant(0.01, shape=[1], name="ATT_WF_3")), 
            }
        print("newnet dimension :", newNet)
        
        imageVec = self.Attention(newNet, self.imageFeature.outputLS, self.textFeature.RNNState, self.extractFeature.finalState, "ATT_img1", "ATT_img2", 196, True)
        textVec = self.Attention(self.textFeature.RNNState, self.textFeature.outputs, newNet, self.extractFeature.finalState, "ATT_text1", "ATT_text2", self.textFeature.seqLen, False)
        attrVec = self.Attention(self.extractFeature.finalState, self.extractFeature.inputEmb, newNet, self.textFeature.RNNState, "ATT_attr1", "ATT_attr2", 5, False)
        
        attHidden = tf.tanh(tf.matmul(imageVec, self.weights["ATT_WI1"])+self.biases["ATT_WI1"])
        attHidden2 = tf.tanh(tf.matmul(textVec, self.weights["ATT_WT1"])+self.biases["ATT_WT1"])
        attHidden3 = tf.tanh(tf.matmul(attrVec, self.weights["ATT_WA1"])+self.biases["ATT_WW"])
        scores1 = tf.matmul(attHidden, self.weights["ATT_WF_1"])+self.biases["ATT_WF_1"]
        scores2 = tf.matmul(attHidden2, self.weights["ATT_WF_2"])+self.biases["ATT_WF_2"]
        scores3 = tf.matmul(attHidden3, self.weights["ATT_WF_3"])+self.biases["ATT_WF_3"]
        scoreLS = [scores1, scores2, scores3]
        scoreLS = tf.nn.softmax(scoreLS, dim=0)
        imageVec = tf.tanh(tf.matmul(imageVec, self.weights["ATT_WI2"])+self.biases["ATT_WI"])
        textVec = tf.tanh(tf.matmul(textVec, self.weights["ATT_WT2"])+self.biases["ATT_WT"])
        attrVec = tf.tanh(tf.matmul(attrVec, self.weights["ATT_WA2"])+self.biases["ATT_WA"])
        self.concatInput = scoreLS[0]*imageVec+scoreLS[1]*textVec+scoreLS[2]*attrVec

    def Attention(self, g0, sequence, g1, g2, AttStr, AttStr2, length, flag):
        tmpLS = []
        tmpLS2 = []
        tmpLS3 = []
        if not flag:
            seq = tf.transpose(sequence, [1, 0, 2])
        else:
            seq = sequence
        for i in range(length):
            nHidden = tf.tanh(tf.matmul(tf.concat([seq[i], g1], axis=1), self.weights[AttStr+"_1"]) + self.biases[AttStr+"_1"])
            nHidden2 = tf.tanh(tf.matmul(tf.concat([seq[i], g2], axis=1), self.weights[AttStr+"_2"]) + self.biases[AttStr+"_2"])
            nHidden3 = tf.tanh(tf.matmul(tf.concat([seq[i], g0], axis=1), self.weights[AttStr+"_3"]) + self.biases[AttStr+"_3"])
            tmpLS.append(tf.matmul(nHidden, self.weights[AttStr2+"_1"]) + self.biases[AttStr2+"_1"])
            tmpLS2.append(tf.matmul(nHidden2, self.weights[AttStr2+"_2"]) + self.biases[AttStr2+"_2"])
            tmpLS3.append(tf.matmul(nHidden3, self.weights[AttStr2+"_3"]) + self.biases[AttStr2+"_3"])
        tmpLS = tf.nn.softmax(tmpLS, dim=0)
        tmpLS2 = tf.nn.softmax(tmpLS2, dim=0)
        tmpLS3 = tf.nn.softmax(tmpLS3, dim=0)
        self.representation_score[AttStr] = (tmpLS+tmpLS2+tmpLS3)/3
        ret = tmpLS[0] * seq[0] / 3 + tmpLS2[0] * seq[0] / 3 + tmpLS3[0] * seq[0] / 3 
        for i in range(1, length):
            ret += tmpLS[i] * seq[i] / 3 + tmpLS2[i] * seq[i] / 3 + tmpLS3[i] * seq[i] / 3 
        return ret

    def predicting(self, rate):
        hidden = tf.nn.relu(tf.matmul(self.concatInput, self.weights["MLP1"]) + self.biases["MLP1"])
        logits = tf.matmul(hidden, self.weights["MLP2"]) + self.biases["MLP2"]
        predictPossibility = tf.nn.sigmoid(logits)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(predictPossibility>0.5, tf.float32) , self.y), tf.float32))
        loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=self.y, logits=logits, pos_weight=rate))
        tv = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'training_variable')
        l2_loss = self.l2_para * tf.reduce_sum([tf.nn.l2_loss(v) for v in tv])
        loss += l2_loss
        return loss, accuracy, predictPossibility
