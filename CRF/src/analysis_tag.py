# coding:utf-8

import codecs
import sys
import os

resultdir = "../dataset/result/"
area = [ "病史特点", "一般项目", "出院情况", "诊疗经过"]

def judge_type(itype):
    if itype == "SIGNS":
        return "症状和体征"
    if itype == "CHECK":
        return "检查和检验"
    if itype == "DISEASE":
        return "疾病和诊断"
    if itype == "TREATMENT":
        return "治疗"
    if itype == "BODY":
        return "身体部位"

if __name__ == '__main__':
    for j in range(4):
        for i in range(301, 401):
            tag_file = resultdir + area[j] + '/' + area[j] + '-' + str(i) +'.crf.txt'
            print(tag_file)
            if not os.path.exists(tag_file):
                continue
            print_file = resultdir + area[j] + '/' + area[j] + '-' + str(i) +'.txt'

            with open(tag_file, encoding='utf-8') as fr:
                lines = fr.readlines()
                fw = open(print_file, 'w', encoding='utf-8')

                cur_pos = 0
                ner = ""
                pos_l = -1
                pos_r = -1
                tmp = 0
                ner_type = ""
                for line in lines:
                    if line == '\n':
                        continue
                    word, tag = line.split()
                    word = word.replace('#', ' ')

                    if (tag[0] == 'B' or tag[0] == 'O') and tmp == 1:
                        ans = ner + ' ' + str(pos_l) + ' ' + str(pos_r) + ' ' + ner_type + ';'
                        fw.write(ans)
                        ner = ""
                        tmp = 0

                    if tmp == 0:
                        if tag[0] == 'B':
                            ner = word
                            pos_l = cur_pos
                            pos_r = cur_pos
                            ner_type = judge_type(tag[2:])
                            tmp = 1
                    elif tag[0] == 'I':
                        ner += word
                        pos_r = pos_r + 1

                    cur_pos += 1
                if tmp == 1:
                    ans = ner + ' ' + str(pos_l) + ' ' + str(pos_r) + ' ' + ner_type + ';'
                    fw.write(ans)
                fw.close()




