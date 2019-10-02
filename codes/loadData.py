import numpy as np
from PIL import Image
import random
import pickle
import re

wordPrefix = "../extract/"
dataPrefix = "../text/"
imagePrefix = "../imageVector2/"

class TextItem():
    def __init__(self, sentence, label):
        self.sentence = sentence
        self.label = label
        self.words = None

class TextIterator():
    def __init__(self, batchSize, seqLen):
        self.batchSize = batchSize
        self.seqLen = seqLen
        self.textData = dict()
        self.trainNum = []
        self.validNum = []
        self.testNum = []
        self.word2id = self.getVocab()
        self.attribute2id = self.getVocabAttr()
        dictExtractWords = self.getExtractDict()
        for i in range(3):
            self.readData(i, dictExtractWords)
        self.batchInd = 0
        self.validInd = 0
        self.testInd = 0
        self.epoch = 0
        self.threshold = int(len(self.trainNum) / self.batchSize)
        print(len(self.trainNum), len(self.validNum), len(self.testNum))
        print("rate: ", self.rate)

    def getExtractDict(self):
        file = open(wordPrefix+"extract_all")
        dic = {}
        for line in file:
            ls = eval(line)
            dic[int(ls[0])] = ls[1:]
        return dic

    def getVocab(self):
        file = open("../words/vocab")
        return pickle.load(file)

    def getVocabAttr(self):
        file = open("../ExtractWords/vocab")
        return pickle.load(file)

    def readData(self, i, dic):
        p = n = 0
        if i == 0:
            file = open(dataPrefix+"train.txt")
            ls = self.trainNum
        elif i == 1:
            file = open(dataPrefix+"valid2.txt")
            ls = self.validNum
        else:
            file = open(dataPrefix+"test2.txt")
            ls = self.testNum
        for line in file:
            lineLS = eval(line)
            tmpLS = lineLS[1].split()
            if "sarcasm" in tmpLS:
                continue
            if "sarcastic" in tmpLS:
                continue
            if "reposting" in tmpLS:
                continue
            if "<url>" in tmpLS:
                continue
            if "joke" in tmpLS:
                continue
            if "humour" in tmpLS:
                continue
            if "humor" in tmpLS:
                continue
            if "jokes" in tmpLS:
                continue
            if "irony" in tmpLS:
                continue
            if "ironic" in tmpLS:
                continue
            if "exgag" in tmpLS:
                continue
            assert int(lineLS[0]) not in self.textData
            ls.append(int(lineLS[0]))
            if i == 0:
                if lineLS[-1] == 1:
                    p += 1
                else:
                    n += 1
            self.textData[int(lineLS[0])] = TextItem(lineLS[1], int(lineLS[-1]))
            self.textData[int(lineLS[0])].words = dic[int(lineLS[0])]
        random.shuffle(ls)
        if i == 0:
            self.rate = float(n) / p

    def nextBatch(self):
        images = []
        retText = np.zeros([self.batchSize, self.seqLen])
        retY = np.zeros([self.batchSize, 1])
        retWords = np.zeros([self.batchSize, 5], dtype='int32')
        for i in range(self.batchSize):
            ID = self.trainNum[self.batchSize*self.batchInd+i]
            textItem = self.textData[ID]
            senLS = textItem.sentence.split()
            minLength = min(self.seqLen, len(senLS))
            for j in range(minLength):
                if senLS[j] in self.word2id:
                    retText[i][j] = self.word2id[senLS[j]]
                else:
                    retText[i][j] = self.word2id["<unk>"]
            retY[i][0] = textItem.label
            image = np.load(imagePrefix+str(ID)+".npy")
            images.append(image)
            for j in range(5):
                if textItem.words[j] in self.attribute2id:
                    retWords[i][j] = self.attribute2id[textItem.words[j]]
                else:
                    retWords[i][j] = self.attribute2id["<unk>"]
        images = np.asarray(images).transpose([1, 0, 2])
        self.batchInd += 1
        if self.batchInd == self.threshold:
            self.batchInd = 0
            self.epoch += 1
            random.shuffle(self.trainNum)
        return retText, images, retWords, retY


    def getValid(self, validLen=None):
        if validLen is None:
            validLen = self.batchSize
        minLen = min(validLen, len(self.validNum) - self.validInd*validLen)
        if minLen <= 0:
            self.validInd = 0
            return None, None, None, None, False
        retText = np.zeros([minLen, self.seqLen])
        retY = np.zeros([minLen, 1])
        retWords = np.zeros([minLen, 5], dtype='int32')
        images = []
        for i in range(minLen):
            ID = self.validNum[validLen*self.validInd+i]
            textItem = self.textData[ID]
            senLS = textItem.sentence.split()
            minLength = min(self.seqLen, len(senLS))
            for j in range(minLength):
                if senLS[j] in self.word2id:
                    retText[i][j] = self.word2id[senLS[j]]
                else:
                    retText[i][j] = self.word2id["<unk>"]
            retY[i][0] = textItem.label
            image = np.load(imagePrefix+str(ID)+".npy")
            images.append(image)
            for j in range(5):
                if textItem.words[j] in self.attribute2id:
                    retWords[i][j] = self.attribute2id[textItem.words[j]]
                else:
                    retWords[i][j] = self.attribute2id["<unk>"]
        images = np.array(images).transpose([1, 0, 2])
        self.validInd += 1
        return retText, images, retWords, retY, True


    def getTest(self, testLen=None):
        if testLen is None:
            testLen = self.batchSize
        minLen = min(testLen, len(self.testNum) - self.testInd*testLen)
        if minLen <= 0:
            self.testInd = 0
            return None, None, None, None, False, None
        retText = np.zeros([minLen, self.seqLen])
        retY = np.zeros([minLen, 1])
        fileNameLS = []
        retWords = np.zeros([minLen, 5], dtype='int32')
        images = []
        for i in range(minLen):
            ID = self.testNum[testLen*self.testInd+i]
            fileNameLS.append(ID)
            textItem = self.textData[ID]
            senLS = textItem.sentence.split()
            minLength = min(self.seqLen, len(senLS))
            for j in range(minLength):
                if senLS[j] in self.word2id:
                    retText[i][j] = self.word2id[senLS[j]]
                else:
                    retText[i][j] = self.word2id["<unk>"]
            retY[i][0] = textItem.label
            image = np.load(imagePrefix+str(ID)+".npy")
            images.append(image)
            for j in range(5):
                if textItem.words[j] in self.attribute2id:
                    retWords[i][j] = self.attribute2id[textItem.words[j]]
                else:
                    retWords[i][j] = self.attribute2id["<unk>"]
        images = np.array(images).transpose([1, 0, 2])
        self.testInd += 1
        return retText, images, retWords, retY, True, fileNameLS
