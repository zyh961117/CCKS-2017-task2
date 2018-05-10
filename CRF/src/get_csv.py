# coding:utf-8

import codecs
import sys
import os
import csv

area = ["一般项目", "诊疗经过", "出院情况", "病史特点"]
resultdir = "../dataset/result/"
print_file = "../dataset/result.csv"

if __name__ == '__main__':
    for j in range(4):
        for i in range(301, 401): 
                result_file = resultdir + area[j] + '/' + area[j] + '-' + str(i) +'.txt'
                if not os.path.exists(result_file):
                    print(result_file)
                    continue
                with open(print_file, 'a', encoding='utf-8') as fw:
                    writer = csv.writer(fw)
                    with open(result_file, encoding='utf-8') as fr:
                        lines = fr.read()
                        lines = lines.strip('\n')
                        lines = lines.replace('#', ' ')
                        writer.writerow([str(i), area[j], lines])
