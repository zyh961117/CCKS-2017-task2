# coding: utf-8

import fio
import codecs
import sys
import os

traindir = "../dataset/train/"
testdir = "../dataset/test/"
resultdir = "../dataset/result/"
area = ["病史特点", "出院情况", "一般项目", "诊疗经过"]

if __name__ == '__main__':
    for i in range(301, 401):
        for j in range(4):
            filename = testdir + area[j] + '/' + area[j] + '-' + str(i) +'.txtoriginal.txt'
            if not os.path.exists(filename):
                continue
            sentences = fio.ReadFileUTF8(filename)
            printfile = resultdir + area[j] + '/' + area[j] + '-' + str(i) +'.segment.txt'
            for sentence in sentences:
                if sentence == u'\r':
                    continue
                sentence = sentence.replace(' ', '#')
                for word in sentence:
                    if word == u'\r':
                        continue
                    fio.WriteFileUTF8(word, printfile)