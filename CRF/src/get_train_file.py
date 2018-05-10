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
            train_file = biotagdir + area[j] + '/' + area[j] + '-' + str(i) +'.biotag.txt'
            result_file = '../dataset/train_data'
            lines = fio.ReadFileUTF8(train_file)
            fio.AddTest(lines, result_file)
            fio.WriteFileUTF8('', result_file)
