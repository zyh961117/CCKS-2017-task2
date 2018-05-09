## 主要命令

### 训练模型命令：
```shell
$ python3 main.py --train=True --clean=True
```

### 测试语料命令：
```shell
$ python3 main.py
```

## 主要文件

#### `ckpt/`
model_checkpoint

#### `data/`
分 `dev_set`、`test_set`、`train_set` 三个数据集，分别是官方提供的总训练数据集的 10%、10%、80%

已经进行了 BIO 标记，BIO 标记的代码在 CRF 的文件夹里


#### `log/`
`train.log` 训练模型时的 log 文件


#### `result/`
`一般项目`、`出院情况`、`病史特点`、`诊疗经过` 官方提供的测试数据集的运行结果 txt 文件

`config.txt` 模型的配置信息

`eval.txt` 评测结果

`ner_predict.utf8` 模型的输出结果文件，供 perl 进行性能评估时使用

`result.csv` 最终测试结果的文件


#### `test/`
官方测试数据集

## 主要代码

`config_file` 模型的配置信息

`maps.pkl` 词向量映射结果

`train.log` 训练的 log 信息

`wiki_100.utf8` 预训练好的 wiki 的词向量文件
 


`conlleval.exe` `conlleval.py` 利用 conlleval + perl 对模型性能进行评估，在训练模型时使用

`data_utils.py` 主要用于解析数据，根据词向量建立 map，解析 BIO 标记并转化为 BIOSE 标记

`loader.py` 加载文件、数据等

`rnncell.py` 由 tf 提供的 CoupledInputForgetGateLSTMCell 类

`utils.py` 公用的方法，如：解析实体类别、输出 log、输出配置、创建文件夹等

`model.py` 模型的类，包含输入嵌入层、Bi-LSTM-CRF层、处理层、逻辑回归层等

`main.py` 主要的训练、测试、运行代码



`get_csv.py` 解析 `result/` 四个文件夹中运行结果的 txt 文件，整合输出 `result.csv`

`final_gold_truth.csv` 官方提供的测试数据的人工标注结果

`ccks-eval.py` 官方提供的评测代码，比较`./result.csv` 和 `./final_gold_truth.csv` 两个文件

`eval.txt` 评测结果，即`ccks-eval.py` 的输出文件

`eval.csv` 评测结果的表格
