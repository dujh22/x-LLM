# 通过对齐语言将大型语言模型外推到非英语

> 注意：本项目为针对原始项目的复现和学习（请参考最后的引用），原始项目位置在https://github.com/NJUNLP/x-LLM

此存储库包含该项目的代码实现，旨在通过构建跨语言的语义对齐来增强非英语语言上的预训练大型语言模型（LLM）。该项目探索了跨语言指令微调和多语言指令微调技术。代码实现基于[斯坦福羊驼](https://github.com/tatsu-lab/stanford_alpaca).

![](./xllama.jpg)

## 要求和安装
若要安装此存储库，请按照下列步骤操作：
```
git clone git@github.com:dujh22/x-LLM.git
cd x-LLM
# 修改environment.yml中的内容
  # name是conda环境的名称
  # prefix是conda环境的路径
conda env create -f environment.yml
```

有关 conda 环境的详细信息，请参阅 environment.yml 文件。

```markdown
environment.yml是用conda命令将环境信息导出备份的文件。
	创建命令如下：
		conda env export > environment.yml
	软件安装时则执行以下命令就可以恢复其运行环境和依赖包：
		conda env create -f environment.yml
注1：.yml文件移植过来的环境只是安装了你原来环境里用conda install等命令直接安装的包，你用pip之类装的东西没有移植过来，需要你重新安装。--待确认。
注2：environment.yml中包含该文件创建时所在的虚拟环境名称，不需要先执行"conda env create"创建并进入虚拟环境，直接在base环境下执行就会自动创建虚拟环境以及安装其中的依赖包（这个是与pip install -r requirements.txt不同的）。当然这就要求你的当前环境中没有同名的虚拟环境。如果暗装者不想使用environment.yml中内置的虚拟环境名(在environment.yml的第一行)，可以使用-n选项来指定新的虚拟环境名，如下所示：
		conda env create -f environment.yml -n new_env_name
```

注意：原始environment.yml在复现过程中产生一些错误，需要根据错误提示进行相应的修改。

可以采用以下手段：

1. 去掉了pip后面所有包的版本号

原因：typing-extensions(==4.7.1) 但多个其他包依赖不同的typing-extensions版本

2. 删除bleurt并自行安装

原因：找不到包满足bleurt==0.0.2

方法：参照[google-research/bleurt: BLEURT is a metric for Natural Language Generation based on transfer learning. (github.com)](https://github.com/google-research/bleurt)手动安装

```shell
pip install --upgrade pip  # ensures that pip is current
git clone https://github.com/google-research/bleurt.git
cd bleurt
pip install .
```

3. 删除tensorrt-libs==8.6.1并自行安装

原因：安装子进程报错

方法：再次使用 pip install tensorrt-libs==8.6.1会发现已经安装了

## 用法
### 下载 预训练 LLM
首先将预训练的 LLM 下载到 ./model 目录中。

### 下载数据集
您可以从此[链接](https://drive.google.com/file/d/1bkejieKDJFDJ45UmQYiY4eeqpGBwj-r-/view?usp=drive_link)下载此项目中使用的所有数据集。下载后，将数据集放在 ./data 目录中。数据集包括：

* 训练数据集
  * 羊驼Alpaca
  * 维基矩阵Wikimatrix
  * 新闻评论Newscommentary
* 评估数据集
  * XQUAD
  * MLQA
  * Flores-101
  * MI-Eval

### 加载原始数据以及指令 
您可以使用提供的脚本 （./data/<dataset>/<dataset.py>） 加载原始数据以及指令。如果要使用新的数据集，则需要实现相应的脚本。加载的数据将具有以下结构：

``` python
datasets.Features(
    {
        "id": datasets.Value("string"),
        "instruction": datasets.Value("string"),
        "input": datasets.Value("string"),
        "output": datasets.Value("string")
    }
)
```

## 指令微调预训练 LLM
若要对预训练的 LLM 进行指令微调，请运行 train.sh 脚本。例如，您可以使用以下命令将 LLaMA-7B 指令微调为 x-LLaMA-7B（中文）：

``` bash
bash script/train.sh llama-7b-hf alpaca_en+alpaca_zh+translation_ncwm_en-zh
```
在此命令中，第一个参数表示要使用的预训练 LLM，第二个参数表示要使用的训练数据。您可以使用 + 来连接多个数据集，训练数据将由 Huggingface 训练器进行洗牌。

训练完成后，经过微调的 LLM 将保存在 ./model/llama-7b-hf.alpaca_en+alpaca_zh+translation_ncwm_en-zh.finetune 中。您可以使用别名来定义更短的名称，更多详情请参见 ./data/alias/alias.json。

## 测试微调后的 LLM
要测试微调后的 LLM，请运行 inference.sh 脚本。例如，您可以使用以下命令在 Flores 数据集上测试经过微调的 LLM：
``` bash
bash script/inference.sh llama-7b-hf.alpaca_en+alpaca_zh+translation_ncwm_en-zh.finetune translation_flores_en-zh
```
输出结果将保存在 model/llama-7b-hf.alpaca_en+alpaca_zh+translation_ncwm_en-zh.finetune/test/translation_flores_en-zh.inference.jsonl 中。预测字段表示 LLM 生成的内容。

## 通过 Web UI 与 LLM 交互

若要通过 Web UI 与 LLM 交互，请使用以下命令运行 app.py：
``` bash
bash app.py model/llama-7b-hf.alpaca_en+alpaca_zh+translation_ncwm_en-zh.finetune
```

## 引文

如果您觉得此存储库有帮助，请考虑引用我们的论文：
```
@misc{zhu2023extrapolating,
      title={Extrapolating Large Language Models to Non-English by Aligning Languages}, 
      author={Wenhao Zhu and Yunzhe Lv and Qingxiu Dong and Fei Yuan and Jingjing Xu and Shujian Huang and Lingpeng Kong and Jiajun Chen and Lei Li},
      year={2023},
      eprint={2308.04948},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```