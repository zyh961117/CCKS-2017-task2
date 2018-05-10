# coding: utf-8

import jieba
import fio
import codecs
import sys
import os
import argparse, time, random

traindir = "../dataset/train/"
biotagdir = "../dataset/biotag/"
area = ["一般项目", "病史特点", "出院情况", "诊疗经过"]

def judge_type(itype):
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

if __name__ == '__main__':
    for i in range(1, 301):
        for j in range(4):
            original_file = traindir + area[j] + '/' + area[j] + '-' + str(i) +'.txtoriginal.txt'
            tag_file = traindir + area[j] + '/' + area[j] + '-' + str(i) +'.txt'
            printfile = biotagdir + area[j] + '/' + area[j] + '-' + str(i) +'.biotag.txt'
            lines = fio.ReadFileUTF8(original_file)
            ners = fio.ReadFileUTF8(tag_file)

            tmp = 0
            if tmp < len(ners):
                cur_ner = ners[tmp].split()
            else:
                cur_ner = ['', -1, -1]
            for words in lines:
                for k in range(len(words)):
                    if str(k) >= cur_ner[1] and str(k) <= cur_ner[2]:
                        ner_type = judge_type(cur_ner[3])
                        if str(k) == cur_ner[1]:
                            ans = words[k] + " B-" + ner_type
                        else:
                            ans = words[k] + " I-" + ner_type
                        fio.WriteFileUTF8(ans, printfile)
                        if str(k) == cur_ner[2]:
                            tmp += 1
                            if tmp < len(ners):
                                cur_ner = ners[tmp].split()
                            else:
                                cur_ner = ['', -1, -1]
                    else:
                        ans = words[k] + " O"
                        fio.WriteFileUTF8(ans, printfile)
