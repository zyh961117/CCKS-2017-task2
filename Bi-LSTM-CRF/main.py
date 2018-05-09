# -*- coding: UTF-8 -*-
# -*- coding: cp936 -*-  
import os
import codecs
import pickle
import itertools
from collections import OrderedDict
from tkinter import *
from tkinter import ttk, Scrollbar, Frame
import threading

import tensorflow as tf
import numpy as np
from model import Model
from loader import load_sentences, update_tag_scheme
from loader import char_mapping, tag_mapping
from loader import augment_with_pretrained, prepare_dataset
from utils import get_logger, make_path, clean, create_model, save_model
from utils import print_config, save_config, load_config, test_ner
from data_utils import load_word2vec, create_input, input_from_line, BatchManager

flags = tf.app.flags
flags.DEFINE_boolean("clean",       False,      "clean train folder") 
flags.DEFINE_boolean("train",       False,      "Wither train the model")
# configurations for the model
flags.DEFINE_integer("seg_dim",     20,         "Embedding size for segmentation, 0 if not used") #句子的最长长度，循环次数
flags.DEFINE_integer("char_dim",    100,        "Embedding size for characters") #输入层（字向量）的长度
flags.DEFINE_integer("lstm_dim",    100,        "Num of hidden units in LSTM") #隐藏层的长度
flags.DEFINE_string("tag_schema",   "iobes",    "tagging schema iobes or iob") #标记方式BIO

# configurations for training
flags.DEFINE_float("clip",          5,          "Gradient clip") #梯度阈值，防止梯度消失和梯度爆炸
flags.DEFINE_float("dropout",       0.5,        "Dropout rate") #丢失的概率，防止过拟合
flags.DEFINE_float("batch_size",    1,          "batch size")
flags.DEFINE_float("lr",            0.001,      "Initial learning rate") #学习率
flags.DEFINE_float("train_step",    100,        "Learning steps") #学习次数
flags.DEFINE_string("optimizer",    "adagrad",     "Optimizer for training") #优化器
flags.DEFINE_boolean("pre_emb",     True,       "Wither use pre-trained embedding") #是否使用已经预训练好的数据wiki-100
flags.DEFINE_boolean("zeros",       False,      "Wither replace digits with zero") 
flags.DEFINE_boolean("lower",       True,       "Wither lower case") #英文的大小写转换

flags.DEFINE_integer("max_epoch",   100,        "maximum training epochs") #训练次数
flags.DEFINE_integer("steps_check", 80,         "steps per checkpoint") #检查点
flags.DEFINE_string("ckpt_path",    "ckpt",     "Path to save model")
flags.DEFINE_string("summary_path", "summary",      "Path to store summaries")
flags.DEFINE_string("log_file",     "train.log",    "File for log")
flags.DEFINE_string("map_file",     "maps.pkl",     "file for maps")
flags.DEFINE_string("vocab_file",   "vocab.json",   "File for vocab")
flags.DEFINE_string("config_file",  "config_file",  "File for config")
flags.DEFINE_string("script",       "conlleval",    "evaluation script")
flags.DEFINE_string("result_path",  "result",       "Path for results")
flags.DEFINE_string("emb_file",     "wiki_100.utf8", "Path for pre_trained embedding")
flags.DEFINE_string("train_file",   os.path.join("data", "train_set"),  "Path for train data")
flags.DEFINE_string("dev_file",     os.path.join("data", "dev_set"),    "Path for dev data")
flags.DEFINE_string("test_file",    os.path.join("data", "test_set"),   "Path for test data")


FLAGS = tf.app.flags.FLAGS
assert FLAGS.clip < 5.1, "gradient clip should't be too much"
assert 0 <= FLAGS.dropout < 1, "dropout rate between 0 and 1"
assert FLAGS.lr > 0, "learning rate must larger than zero"
assert FLAGS.optimizer in ["adam", "sgd", "adagrad"]

area = ["一般项目", "诊疗经过", "出院情况", "病史特点"]
testdir = "./test/"
resultdir = "./final_result/"


# config for the model
def config_model(char_to_id, tag_to_id):
    config = OrderedDict()
    config["num_chars"] = len(char_to_id)
    config["char_dim"] = FLAGS.char_dim
    config["num_tags"] = len(tag_to_id)
    config["seg_dim"] = FLAGS.seg_dim
    config["lstm_dim"] = FLAGS.lstm_dim
    config["batch_size"] = FLAGS.batch_size
    config["train_step"] = FLAGS.train_step
    config["emb_file"] = FLAGS.emb_file
    config["clip"] = FLAGS.clip
    config["dropout_keep"] = 1.0 - FLAGS.dropout
    config["optimizer"] = FLAGS.optimizer
    config["lr"] = FLAGS.lr
    config["tag_schema"] = FLAGS.tag_schema
    config["pre_emb"] = FLAGS.pre_emb
    config["zeros"] = FLAGS.zeros
    config["lower"] = FLAGS.lower
    config["max_epoch"] = FLAGS.max_epoch
    config["steps_check"] = FLAGS.steps_check
    return config


def evaluate(sess, model, name, data, id_to_tag, logger):
    logger.info("evaluate:{}".format(name))
    ner_results = model.evaluate(sess, data, id_to_tag)
    eval_lines = test_ner(ner_results, FLAGS.result_path)
    for line in eval_lines:
        logger.info(line)
    f1 = float(eval_lines[1].strip().split()[-1])

    if name == "dev":
        best_test_f1 = model.best_dev_f1.eval()
        if f1 > best_test_f1:
            tf.assign(model.best_dev_f1, f1).eval()
            logger.info("new best dev f1 score:{:>.3f}".format(f1))
        return f1 > best_test_f1
    elif name == "test":
        best_test_f1 = model.best_test_f1.eval()
        if f1 > best_test_f1:
            tf.assign(model.best_test_f1, f1).eval()
            logger.info("new best test f1 score:{:>.3f}".format(f1))
        return f1 > best_test_f1


def train():
    # 映射数据集
    # load data sets
    train_sentences = load_sentences(FLAGS.train_file, FLAGS.lower, FLAGS.zeros)
    dev_sentences = load_sentences(FLAGS.dev_file, FLAGS.lower, FLAGS.zeros)
    test_sentences = load_sentences(FLAGS.test_file, FLAGS.lower, FLAGS.zeros)

    # Use selected tagging scheme (IOB / IOBES)
    update_tag_scheme(train_sentences, FLAGS.tag_schema)
    update_tag_scheme(test_sentences, FLAGS.tag_schema)

    # create maps if not exist
    if not os.path.isfile(FLAGS.map_file):
        # create dictionary for word
        if FLAGS.pre_emb:
            dico_chars_train = char_mapping(train_sentences, FLAGS.lower)[0]
            dico_chars, char_to_id, id_to_char = augment_with_pretrained(
                dico_chars_train.copy(),
                FLAGS.emb_file,
                list(itertools.chain.from_iterable(
                    [[w[0] for w in s] for s in test_sentences])
                )
            )
        else:
            _c, char_to_id, id_to_char = char_mapping(train_sentences, FLAGS.lower)

        # Create a dictionary and a mapping for tags
        _t, tag_to_id, id_to_tag = tag_mapping(train_sentences)
        with open(FLAGS.map_file, "wb") as f:
            pickle.dump([char_to_id, id_to_char, tag_to_id, id_to_tag], f)
    else:
        with open(FLAGS.map_file, "rb") as f:
            char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)

    # prepare data, get a collection of list containing index
    train_data = prepare_dataset(
        train_sentences, char_to_id, tag_to_id, FLAGS.lower
    )
    dev_data = prepare_dataset(
        dev_sentences, char_to_id, tag_to_id, FLAGS.lower
    )
    test_data = prepare_dataset(
        test_sentences, char_to_id, tag_to_id, FLAGS.lower
    )
    print("%i / %i / %i sentences in train / dev / test." % (
        len(train_data), 0, len(test_data)))

    train_manager = BatchManager(train_data, FLAGS.batch_size)
    dev_manager = BatchManager(dev_data, 100)
    test_manager = BatchManager(test_data, 100)

    # 配置文件路径
    make_path(FLAGS)
    if os.path.isfile(FLAGS.config_file):
        config = load_config(FLAGS.config_file)
    else:
        config = config_model(char_to_id, tag_to_id)
        save_config(config, FLAGS.config_file)
    make_path(FLAGS)

    log_path = os.path.join("log", FLAGS.log_file)
    logger = get_logger(log_path)
    print_config(config, logger)

    # limit GPU memory
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    # 训练模型
    steps_per_epoch = train_manager.len_data
    with tf.Session(config=tf_config) as sess:
        model = create_model(sess, Model, FLAGS.ckpt_path, load_word2vec, config, id_to_char, logger)
        logger.info("start training")
        loss = []
        for i in range(FLAGS.train_step):
            for batch in train_manager.iter_batch(shuffle=True):
                step, batch_loss = model.run_step(sess, True, batch)
                loss.append(batch_loss)
                if step % FLAGS.steps_check == 0:
                    iteration = step // steps_per_epoch + 1
                    logger.info("iteration:{} step:{}/{}, "
                                "NER loss:{:>9.6f}".format(
                        iteration, step%steps_per_epoch, steps_per_epoch, np.mean(loss)))
                    loss = []

            best = evaluate(sess, model, "dev", dev_manager, id_to_tag, logger)
            if best:
                save_model(sess, model, FLAGS.ckpt_path, logger)
            evaluate(sess, model, "test", test_manager, id_to_tag, logger)

global m_model
global m_sess

def evaluate_line():
    config = load_config(FLAGS.config_file)
    logger = get_logger(FLAGS.log_file)
    # limit GPU memory
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with open(FLAGS.map_file, "rb") as f:
        char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)
    global m_model
    global m_sess
    m_sess = tf.Session(config=tf_config)
    m_model = create_model(m_sess, Model, FLAGS.ckpt_path, load_word2vec, config, id_to_char, logger)

    # with tf.Session(config=tf_config) as sess:
    #     m_model = create_model(sess, Model, FLAGS.ckpt_path, load_word2vec, config, id_to_char, logger)
        # while True:
        #     line = input("请输入测试句子:")
        #     result = model.evaluate_line(sess, input_from_line(line, char_to_id), id_to_tag)
        #     print(result)
        # for i in range(301, 401):
        #     for j in range(4):
        #         test_file = testdir + area[j] + '/' + area[j] + '-' + str(i) +'.txtoriginal.txt'
        #         result_file = resultdir + area[j] + '/' + area[j] + '-' + str(i) +'.txt'
        #         if not os.path.exists(test_file):
        #             continue
        #         with open(test_file, encoding='utf-8') as fr:
        #             line = fr.read()
        #             line = line.strip('\n')
        #             result = model.evaluate_line(sess, input_from_line(line, char_to_id), id_to_tag)
        #             fw = open(result_file, 'w', encoding='utf-8')
        #             fw.write(result)
        #             fw.close()

global entry
global tree

# 测试输入数据
def runTest():
    global entry
    test_data = entry.get("0.0", "end")
    
    config = load_config(FLAGS.config_file)
    logger = get_logger(FLAGS.log_file)
    # limit GPU memory
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with open(FLAGS.map_file, "rb") as f:
        char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)
    global m_model
    global m_sess
    with m_sess.as_default(): 
        line = test_data
        result = m_model.evaluate_line(m_sess, input_from_line(line, char_to_id), id_to_tag)
        showResult(result)

# 显示结果
def showResult(result):
    global tree
    items = tree.get_children()
    [tree.delete(item) for item in items]

    ners = result.split(';')
    for i in range(len(ners)):
        if(len(ners[i].split()) != 4):
            continue
        name, pos1, pos2, ner_type = ners[i].split()
        tree.insert("", i, values=(name, pos1, pos2, ner_type))


# 文本全选
def selectText(event):
    entry.tag_add(SEL, "1.0", END)
    return 'break'

def show_window():
    evaluate_line()

    root = Tk()

    root.title("中文电子病历的命名实体识别系统")

    # 设置窗口大小
    width, height = 600, 600

    # 窗口居中显示
    root.geometry('%dx%d+%d+%d' % (width,height,(root.winfo_screenwidth() - width ) / 2, (root.winfo_screenheight() - height) / 2))

    frame = Frame(root)
    frame.place(x=130, width=350, height=500)

    Label(frame, height=1, text="请输入测试句子:", compound='left').pack()

    global entry    
    entry = Text(frame, height=7, width=50)
    entry.pack()
    # 绑定全选快捷键
    entry.bind_all("<Command-a>", selectText)

    # command绑定回调函数
    button = Button(frame, text="Run", command=runTest)
    button.pack()

    scrollBar = Scrollbar(frame)
    scrollBar.pack(side=RIGHT, fill=Y)

    global tree
    tree = ttk.Treeview(frame, columns=("名称","起点","终点","类型"), show="headings", yscrollcommand=scrollBar.set)

    tree.column("名称", width=150, anchor='center')
    tree.column("起点", width=50, anchor='center')
    tree.column("终点", width=50, anchor='center')
    tree.column("类型", width=80, anchor='center')

    tree.heading("名称", text="命名实体名称")
    tree.heading("起点", text="起始位置")
    tree.heading("终点", text="结束位置")
    tree.heading("类型", text="命名实体类型")
    tree.pack(side=LEFT, fill=Y) 

    scrollBar.config(command=tree.yview)
    # 进入消息循环
    root.mainloop()
    

def main(_):

    if FLAGS.train:
        if FLAGS.clean:
            clean(FLAGS)
        train()
    else:
        show_window()


if __name__ == "__main__":
    tf.app.run(main)

