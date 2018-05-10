# coding:utf-8

import fio
import codecs
import sys
import os

datadir = "../data/trainingset 1-100"
area = ["病史特点", "出院情况", "一般项目", "诊疗经过"]

class CRF_unit:
    def __init__(self):
        self.features = []

    def test_into_aline(self, filename):
        self.features = []
        sentences = fio.ReadFileUTF8(filename)
        for sentence in sentences:
            for token in sentence:
                self.features.append(token)

    def get_token(self, filename):
        self.features = []
        sentences = fio.ReadFileUTF8(filename)
        for sentence in sentences:
            for token in sentence:
                feature = [token, "N"]
                self.features.append(feature)

    def read_type(self, itype):
        itype = itype.encode('utf-8')
        if itype == "症状和体征":
            return "SIGNS"
        if itype == "检查和检验":
            return "CHECK"
        if itype == "疾病和诊断":
            return "DISEASE"
        if itype == "治疗":
            return "TREATMENT"
        if itype == "身体部位":
            return "BODY"


    def get_type(self, filename):
        sentences = fio.ReadFileUTF8(filename);
        for sentence in sentences:
            words = sentence.split()
            x = int(words[1])
            y = int(words[2])

            #if words[3].encode('utf-8') == "身体部位":
            itype = self.read_type(words[3])
            self.features[x][1] = "B-" + itype
            for j in range(x+1,y+1):
                self.features[j][1] = "I-" + itype

if __name__ == '__main__':
    extractor = CRF_unit()
    x = 0;
    # for i in range(1,91):
    #     filename = datadir + '/' + area[x] + '/' + area[x] + '-'+ str(i) +'.txtoriginal.txt'
    #     extractor.get_token(filename)

    #     filename = datadir + '/' + area[x] + '/' + area[x] + '-'+ str(i) +'.txt'
    #     extractor.get_type(filename)

    #     filename = datadir + '/result/' + area[x] + '.body_train.txt'
    #     fio.AddTrain(extractor.features, filename)

    for i in range(91, 101):
        filename = datadir + '/' + area[x] + '/' + area[x] + '-'+ str(i) +'.txtoriginal.txt'
        extractor.test_into_aline(filename);

        filename = datadir + '/result/' + area[x] + '.test-' + str(i) + '.txt'
        fio.AddTest(extractor.features, filename)
