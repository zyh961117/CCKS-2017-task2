# coding:utf-8
import sys
import os
import codecs

datadir = "../data/trainingset 1-100"

def ReadFile(file):

    f = open(file,'r')
    lines = f.readlines()
    f.close()
    
    return [line.rstrip('\r\n') for line in lines]

def ReadFileUTF8(file):

    if os.path.exists(file):
        f = codecs.open(file,'r', 'utf8')
        lines = f.readlines()
        f.close()
        
        return [line.rstrip('\n') for line in lines]
    else:
        return ''

def WriteFileUTF8(word, file, linetag="\n"):

    reload(sys)
    sys.setdefaultencoding('utf8')
    
    f = open(file, "a")
    f.write(word + linetag)
    f.flush()
    f.close()

def SaveDict(dict, file, linetag="\n"):
    
    reload(sys)
    sys.setdefaultencoding('utf8')
    
    f = open(file, "w")
    for key, value in dict.items():
        f.write(key + " " + str(value))
        f.write(linetag)
    f.flush()
    f.close()

def SaveFeatures(features, file, linetag="\n"):

    reload(sys)
    sys.setdefaultencoding('utf8')
    
    f = open(file, "w")
    for [token,tag] in features:
        f.write(token + " " + tag)
        f.write(linetag)
    f.flush()
    f.close()

def AddTrain(features, file, linetag="\n"):
    
    reload(sys)
    sys.setdefaultencoding('utf8')
    
    f = open(file, "a")
    for [token,tag] in features:
        f.write(token + " " + tag)
        f.write(linetag)
    f.flush()
    f.close()

def AddTest(tokens, file, linetag="\n"):
    
    reload(sys)
    sys.setdefaultencoding('utf8')
    
    f = open(file, "a")
    for token in tokens:
        f.write(token)
        f.write(linetag)
    f.flush()
    f.close()

def AddResult(total, file, linetag="\n"):
    
    reload(sys)
    sys.setdefaultencoding('utf8')
    
    f = open(file, "a")
    for name, left, right, itype in total:
        f.write(name + "\t" + str(left) + "\t" + str(right) + "\t" + itype)
        f.write(linetag)
    f.flush()
    f.close()

if __name__ == '__main__':
    lines = ReadFileUTF8(datadir+'/病史特点/病史特点-1.txtoriginal.txt');
    for line in lines:
        print line