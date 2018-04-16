# -*- coding: UTF-8 -*-

from tkinter import *
from main import test_evaluate

def runTest():
    test_data = entry.get()
    text.delete(0.0, END)
    test_evaluate()
    # text.insert(END, test_evaluate(test_data))

root = Tk()

root.title("Chinese NER")

# 设置窗口大小
width, height = 600, 500

# 窗口居中显示
root.geometry('%dx%d+%d+%d' % (width,height,(root.winfo_screenwidth() - width ) / 2, (root.winfo_screenheight() - height) / 2))

entry = Entry(root)
entry.pack()

# command绑定回调函数
button = Button(root, text="Run", command=runTest)
button.pack()

text = Text(root, height=4, width=100)
text.pack()

# 进入消息循环
root.mainloop()