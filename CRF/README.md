### `CRF++-0.58/`

CRF++ 的工具包

### `dataset/`

#### `biotag/`

BIO 标记后的训练数据集

#### `result/`

`***.segment.txt` 测试数据集分字处理后的文件

`***.crf.txt` 经过 CRF 模型测试后的结果

`***.txt` 处理成要求的输出格式

`result.csv` 所有测试数据集的整合

`eval.txt` 评测结果

#### `crf_model` 训练好的 crf 模型

#### `template` 特征模板

#### `train.sh` 训练模型的命令

#### `test.sh` 测试语料的命令

#### `train_data` 训练数据的整合

#### `final_gold_truth.csv` 官方提供的测试数据的人工标注结果

### `src/`

`analysis_tag.py` 解析 tag，将 `***.crf.txt` 转化为 `***.txt`

`crf_unit.py` crf 类，主要用于判断命名实体类型以及数据转化为 crf++ 所需的输入格式

`fio.py` 文件读写的相关方法

`get_BIOtag.py` 用于 BIO 标记

`get_csv.py` 解析 `result/` 四个文件夹中运行结果的 `***.txt` 文件，整合输出 `result.csv`

`get_train_file.py` 根据 BIO 标记文件整合得到 `result/train_data`

`get_test_file.py` 得到 `result/***.segment.txt` 文件

`ccks-eval.py` 官方提供的评测代码，比较`result.csv` 和 `final_gold_truth.csv` 两个文件