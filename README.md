# 反诈骗短信识别（RoBERTa 中文文本分类）

用 RoBERTa 中文预训练模型做全参数微调，判断一条短信是不是诈骗，以及具体是哪一类诈骗。项目里有两个模型配合使用：先判断"正常还是可疑"，如果可疑，再判断它属于哪一种诈骗。

## 能做什么

- **二分类**：判断短信是"正常"还是"可疑"。
- **多分类**：对可疑短信进一步分类，一共 5 类：营销骚扰、冒充公检法类、贷款类、冒充客服类、冒充领导熟人类。
- **两种使用方式**：可以在命令行里输入短信看结果，也可以打开本地网页，把短信粘贴进去点按钮判断。

## 项目结构

```
.
├── data/
│   ├── holdout/          # 独立测试数据（不参与训练，用来检验真实效果）
│   ├── processed/        # 清洗后的完整数据和标签映射
│   └── splits_cn2/       # 划分好的训练集、验证集、测试集
├── models/               # 训练好的模型（文件较大，需从 Releases 下载）
├── src/
│   ├── train_cn2.py      # 模型训练脚本
│   ├── predict.py        # 命令行交互预测
│   ├── web_app.py        # 网页版演示（Gradio）
│   ├── eval_v2.py        # 模型效果评估
│   ├── build_v2.py       # 数据处理
│   └── augment_data.py   # 数据增强
└── Telecom_Fraud_Texts_5-main/   # 原始短信数据
```

## 数据规模

| 任务 | 训练集 | 验证集 | 测试集 |
|------|--------|--------|--------|
| 二分类 | 9444 条 | 1186 条 | 1180 条 |
| 多分类 | 7514 条 | 934 条 | 934 条 |

## 环境准备

需要 Python 3.9 以上，然后安装下面这些库：

```bash
pip install torch transformers datasets pandas scikit-learn gradio numpy
```

有 NVIDIA 显卡会快很多，没有显卡也能跑，只是训练和预测慢一些。训练脚本做了显存适配，4GB 显存就能跑。

## 下载模型

模型文件比较大（每个约 390MB），没有放在代码里，需要单独下载：

1. 打开 [Releases 页面](https://github.com/xxxmark-x/Anti-fraud/releases)
2. 下载 `roberta_binary_cn2.zip` 和 `roberta_type_cn2.zip`
3. 分别解压，把解压出来的两个文件夹放到项目的 `models/` 目录下

放好后结构应该是：

```
models/
├── roberta_binary_cn2/
└── roberta_type_cn2/
```

## 重要：先改路径

代码里的项目路径是写死的，运行前需要改成你自己电脑上的实际路径。

打开 `src/` 下的 `.py` 文件，找到开头这一行：

```python
BASE = r"C:\Users\31887\Desktop\反诈骗短信模型2"
```

把引号里的路径改成你这个项目文件夹的实际位置，比如：

```python
BASE = r"C:\Users\你的用户名\Desktop\RoBERTa 预训练模型+全参数微调"
```

`predict.py`、`web_app.py`、`train_cn2.py`、`eval_v2.py` 里都有这一行，都要改。

## 怎么使用

### 方式一：网页版（推荐）

```bash
cd src
python web_app.py
```

运行后会自动打开浏览器，把短信粘贴进去，点"开始判断"就能看到结果。模型只在你自己电脑上运行，短信内容不会传到外面。

### 方式二：命令行

```bash
cd src
python predict.py
```

把短信内容粘贴进去回车，就能看到判断结果。输入 `q` 回车退出。

## 自己训练模型

如果想重新训练，先确认数据已经放在 `data/splits_cn2/` 下，然后运行：

```bash
cd src

# 训练二分类模型
python train_cn2.py --task binary_cn2

# 训练多分类模型
python train_cn2.py --task type_cn2
```

训练的主要设置：

- 基础模型：`hfl/chinese-roberta-wwm-ext`（中文 RoBERTa 全词掩码模型）
- 训练轮数：4 轮
- 学习率：2e-5
- 每条短信最大长度：128 字
- batch size：4，梯度累积 4 步（等效 batch size 16）
- 训练中自动保存效果最好的模型，按 macro F1 分数挑选

如果训练中途断了，可以加 `--resume` 从上次的 checkpoint 接着训：

```bash
python train_cn2.py --task binary_cn2 --resume
```

## 评估模型效果

```bash
cd src
python eval_v2.py
```

会在测试集和独立 holdout 数据上分别评估，输出准确率、混淆矩阵和每一类的精确率、召回率、F1 分数，还会用几条示例短信做演示。

## 标签说明

**二分类标签：**

| 标签 | 含义 |
|------|------|
| 正常 | 正常短信 |
| 可疑 | 营销或诈骗短信 |

**多分类标签：**

| 标签 | 含义 |
|------|------|
| 营销骚扰 | 广告推销、营销骚扰短信 |
| 冒充公检法类 | 冒充公检法及政府机关 |
| 贷款类 | 贷款、代办信用卡 |
| 冒充客服类 | 冒充客服服务 |
| 冒充领导熟人类 | 冒充领导、熟人 |

## 判断逻辑

模型先用二分类模型判断短信的可疑程度。如果可疑概率低于 50%，直接判定为正常；如果达到或超过 50%，再交给多分类模型判断具体类型。这个 50% 的门槛在代码里可以自己调整。
