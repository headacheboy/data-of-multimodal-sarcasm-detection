import loadData
import NNManager
import tensorflow as tf
import time

class Main():
    def __init__(self):
        self.init_lrate = 0.001
        self.nHidden = 256 
        self.seqLen = 75 
        self.batchSize = 32
        self.maxClipping = 5
        self.pKeep = 0.8
        self.maxEpoch = 8 
        self.displayStep = 100 
        self.former = 0.0
        self.patience = 5
        self.maxPatience = 5
        self.maxTF1_validF1 = 0 
        self.globalStep = tf.Variable(0, trainable=False)
        self.maxF1 = self.pre = self.rec = self.acc = self.maxTF1 = self.maxTpre =self.maxTrec = self.maxTacc = 0
        self.texti = loadData.TextIterator(self.batchSize, self.seqLen)
        self.validStep = int(self.texti.threshold/3)
        self.lrate = tf.train.exponential_decay(self.init_lrate, global_step=self.globalStep, decay_steps=self.texti.threshold, decay_rate=0.8)
        self.addGlobal = self.globalStep.assign_add(1)
        self.net = NNManager.Net(self.nHidden, self.seqLen)
        self.loss, self.PredAcc, self.p = self.net.predicting(self.texti.rate)
        updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        tVars = tf.trainable_variables()
        gradients, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tVars), self.maxClipping)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lrate)
        self.trainOP = optimizer.apply_gradients(zip(gradients, tVars))
        self.fileNameOutput = []
        self.start = time.time()

    def getScore(self, p, y, fileNameLS = None):
        tp = fp = tn = fn = 0
        for i in range(p.shape[0]):
            if y[i][0] == 1:
                if p[i][0] > 0.5:
                    tp += 1
                else:
                    if fileNameLS is not None:
                        self.fileNameOutput.append(fileNameLS[i])
                    fn += 1
            else:
                if p[i][0] > 0.5:
                    if fileNameLS is not None:
                        self.fileNameOutput.append(fileNameLS[i])
                    fp += 1
                else:
                    tn += 1
        return tp, fp, tn, fn

    def getF1(self, tp, fp, tn, fn):
        try:
            pre = float(tp) / (tp+fp)
            rec = float(tp) / (tp+fn)
            f1 = 2*pre*rec / (pre+rec)
        except:
            pre = rec = f1 = 0
        return pre, rec, f1

    def display(self, cost, acc, p, y, epochNum):
        tp, fp, tn, fn = self.getScore(p, y)
        pre, rec, f1 = self.getF1(tp, fp, tn, fn)
        end = time.time()
        print("Epoch {0}, f1={1:.5f}, pre={2:.5f}, rec={3:.5f}, acc={4:.5f}, "
              "time={5:.5f}, loss={6:.8f}".format(epochNum, f1, pre, rec, acc, end-self.start, cost))
        self.start = end

    def valid(self, sess, step):
        tp = fp = tn = fn = 0
        while True:
            validX, validImage, validWords, validY, flag = self.texti.getValid()
            if flag is False:
                break
            cost, _acc, p = sess.run([self.loss, self.PredAcc, self.p], feed_dict=self.make_feed_dict(validX, validImage, validWords, validY, False))
            tmpLS = self.getScore(p, validY)
            tp += tmpLS[0]
            fp += tmpLS[1]
            tn += tmpLS[2]
            fn += tmpLS[3]
        print("tp, fp, tn, fn", tp, fp, tn, fn)
        pre, rec, f1 = self.getF1(tp, fp, tn, fn)
        acc = float(tp+tn) / (tp+fp+tn+fn)
        if f1 <= self.former:
            self.patience -= 1
        else:
            self.patience = self.maxPatience
        self.former = f1
        print("Valid, f1={0:.5f}, pre={1:.5f}, rec={2:.5f}, acc={3:.5f}, paitience={4:d}".format(
            f1, pre, rec, acc, self.patience
        ))
        if f1 >= self.maxF1:
            self.saver.save(sess, "../model/"+str(step)+".ckpt")
            validF1 = f1
            tmpFlag = True
            self.maxF1 = f1
            self.pre = pre
            self.rec = rec
            self.acc = acc
            tp = fp = tn = fn = 0
            while True:
                testX, testImage, testWords, testY, flag, fileNameLS = self.texti.getTest()
                if flag is False:
                    break
                cost, _acc, p = sess.run([self.loss, self.PredAcc, self.p], feed_dict=self.make_feed_dict(testX, testImage, testWords, testY, False))
                tmpLS = self.getScore(p, testY, fileNameLS)
                tp += tmpLS[0]
                fp += tmpLS[1]
                tn += tmpLS[2]
                fn += tmpLS[3]
            print("tp, fp, tn, fn", tp, fp, tn, fn)
            pre, rec, f1 = self.getF1(tp, fp, tn, fn)
            acc = float(tp + tn) / (tp + fp + tn + fn)
            if f1 > self.maxTF1:
                fileO1 = open("../output/wrongList", "w")
                fileO1.write(str(self.fileNameOutput))
                self.maxTF1 = f1
                self.maxTpre = pre
                self.maxTrec = rec
                self.maxTacc = acc
                self.maxTF1_validF1 = validF1
                self.saver.save(sess, "../best_model/best.ckpt")
            self.fileNameOutput = []
            print("test, f1={0:.5f}, pre={1:.5f}, rec={2:.5f}, acc={3:.5f}".format(
                f1, pre, rec, acc
            ))
            fileO = open("../output/test_result", "w")
            fileO.write(str([self.maxTF1, self.maxTpre, self.maxTrec, self.maxTacc]))
            fileO.close()

    def make_feed_dict(self, X, Img, Word, Y, isTraining):
        retFeedDict = {}
        retFeedDict[self.net.y] = Y
        retFeedDict[self.net.textFeature.X] = X
        retFeedDict[self.net.extractFeature.x] = Word
        retFeedDict[self.net.imageFeature.batchSize] = X.shape[0]
        if isTraining:
            retFeedDict[self.net.textFeature.pKeep] = self.pKeep 
        else:
            retFeedDict[self.net.textFeature.pKeep] = 1.0
        for i in range(196):
            retFeedDict[self.net.imageFeature.XList[i]] = Img[i]
        return retFeedDict

    def trainingProcess(self):
        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        avgLoss = 0.0 
        self.saver = tf.train.Saver()
        with tf.Session(config=config) as sess:
            print("init...")
            sess.run(init)
            print("init finished!")
            step = 0
            while self.texti.epoch < self.maxEpoch:
                batchX, batchImage, batchWord, batchY = self.texti.nextBatch()
                
                if step % self.displayStep == 0:
                    acc, p = sess.run([self.PredAcc, self.p], feed_dict=self.make_feed_dict(batchX, batchImage, batchWord, batchY, False))
                    self.display(avgLoss/self.displayStep, acc, p, batchY, self.texti.epoch)
                    avgLoss = 0.0

                if step % self.validStep == 0 and self.texti.epoch > 0:
                    self.valid(sess, step)
                    if self.patience == 0:
                        break

                _, tmpLoss = sess.run([self.trainOP, self.loss], feed_dict=self.make_feed_dict(batchX, batchImage, batchWord, batchY, True))
                avgLoss += tmpLoss

                step += 1
                if self.texti.epoch > -1:
                    sess.run(self.addGlobal)


if __name__ == '__main__':
    m = Main()
    m.trainingProcess()