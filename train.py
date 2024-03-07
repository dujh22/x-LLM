#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    根据 Apache 许可 2.0 版（"许可"）授权；
#    除非遵守许可协议，否则不得使用本文件。
#    您可以从以下网址获取许可证副本
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    除非适用法律要求或书面同意，否则根据本许可证分发的软件 
#    均以 "原样 "为基础分发，不提供任何明示或暗示的保证或条件。
#    有关许可协议下的权限和限制的具体语言，请参阅许可协议。


import copy # copy包用于复制对象
import logging # logging模块用于记录日志
from dataclasses import dataclass, field # dataclasses模块用于创建数据类
from typing import Dict, Optional, Sequence # typing模块用于类型提示

import torch # torch是一个开源的机器学习库，用于自然语言处理
import transformers # transformers是一个用于自然语言处理的库
import utils # utils是一个用于处理数据的库
from torch.utils.data import Dataset # torch.utils.data是一个用于处理数据的库
from transformers import Trainer # transformers是一个用于自然语言处理的库

# 定义忽略索引
IGNORE_INDEX = -100 # 定义忽略索引
DEFAULT_PAD_TOKEN = "<pad>" # 定义默认填充标记
DEFAULT_EOS_TOKEN = "</s>"  # 定义默认结束标记
DEFAULT_BOS_TOKEN = "<s>"   # 定义默认开始标记
DEFAULT_UNK_TOKEN = "<unk>" # 定义默认未知标记
# 定义提示字典
PROMPT_DICT = {
    # 定义提示输入
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. " # 下面是描述任务的指令，配对的输入提供了更多的上下文。
        "Write a response that appropriately completes the request.\n\n" # 编写一个响应，以适当地完成请求。
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:" # 指令：{instruction} 输入：{input} 响应：
    ),
    # 定义提示无输入
    "prompt_no_input": (
        "Below is an instruction that describes a task. " # 下面是描述任务的指令。
        "Write a response that appropriately completes the request.\n\n" # 编写一个响应，以适当地完成请求。
        "### Instruction:\n{instruction}\n\n### Response:" # 指令：{instruction} 响应：
    ),
}


# 定义模型参数
@dataclass
class ModelArguments:
    # 模型名称或路径
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


# 定义数据参数
@dataclass
class DataArguments: 
    # 数据路径
    data_path: str = field(default=None, metadata={"help": "Path to the training data."}) # 训练数据的路径


# 定义训练参数
@dataclass
class TrainingArguments(transformers.TrainingArguments): 
    # 缓存目录
    cache_dir: Optional[str] = field(default=None)
    # 输出目录
    optim: str = field(default="adamw_torch")
    # 模型最大长度
    model_max_length: int = field( 
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )

# 定义智能分词器和嵌入调整
def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict, # 特殊标记字典
    tokenizer: transformers.PreTrainedTokenizer, # 预训练分词器
    model: transformers.PreTrainedModel, # 预训练模型
):
    """Resize tokenizer and embedding. 调整标记符和嵌入的大小。

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64. 这是未优化的版本，可能会使您的嵌入大小不能被64整除。
    """
    # 添加特殊标记
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    # num_new_tokens是添加的特殊标记的数量
    # 调整标记符的大小
    model.resize_token_embeddings(len(tokenizer))

    # 如果有新的标记符
    if num_new_tokens > 0:
        # 获取输入嵌入的权重
        input_embeddings = model.get_input_embeddings().weight.data
        # get_input_embeddings()方法用于获取输入嵌入的权重
        # weight.data属性用于获取权重的数据
        # 获取输出嵌入的权重
        output_embeddings = model.get_output_embeddings().weight.data

        # 计算输入嵌入的平均值
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True) # 具体计算方法是将输入嵌入的前num_new_tokens个标记符去掉，然后求平均值
        # dimension=0表示按列求平均值，keepdim=True表示保持原有的维度
        # 计算输出嵌入的平均值
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True) # 具体计算方法是将输出嵌入的前num_new_tokens个标记符去掉，然后求平均值

        # 将输入嵌入的后num_new_tokens个标记符替换为输入嵌入的平均值
        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        # 为什么将输入嵌入的后num_new_tokens个标记符替换为输入嵌入的平均值？因为添加了新的标记符，所以需要调整输入嵌入的大小。前文num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)后 input_embeddings[-num_new_tokens:] 和 output_embeddings[-num_new_tokens:] 都是指向新添加的标记对应的嵌入向量。

        # 将输出嵌入的后num_new_tokens个标记符替换为输出嵌入的平均值
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

# 定义分词函数
# Sequence[str]表示字符串序列
def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings. 分词一个字符串列表。"""
    tokenized_list = [
        # tokenizer()方法用于分词，返回的是一个字典，包括input_ids、attention_mask、token_type_ids等字段
        tokenizer(
            text,                                   # 文本
            return_tensors="pt",                    # 返回张量 # pt表示PyTorch张量
            padding="longest",                      # 填充 # longest表示填充到最长的序列
            max_length=tokenizer.model_max_length,  # 最大长度
            truncation=True,                        # 截断
        )
        for text in strings                         # 遍历字符串
    ]
    # 获取输入标记符
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    # tokenizer.input_ids用于获取输入标记符
    # 获取输入标记符长度
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list # 具体计算方法是将输入标记符中不等于填充标记的标记符求和
        # tokenized.input_ids.ne(tokenizer.pad_token_id)：这是一个比较操作，对 tokenized.input_ids 中的每个元素，检查它是否不等于 tokenizer.pad_token_id（即，检查它是否不是填充标记）。这将返回一个布尔值的张量，其中 True 表示对应的元素不是填充标记，False 表示对应的元素是填充标记。
        # input_ids.ne() 用于获取不等于填充标记的标记符
        # tokenizer.pad_token_id用于获取填充标记
        # input_ids.ne(tokenizer.pad_token_id)用于获取不等于填充标记的标记符
        # sum().item()用于求和
        # item()用于获取标量
    ]
    return dict(
        input_ids=input_ids, # 输入标记符
        labels=labels, # 标签
        input_ids_lens=input_ids_lens, # 输入标记符长度
        labels_lens=labels_lens, # 标签长度
    )

# 预处理
def preprocess( 
    sources: Sequence[str], # 源
    targets: Sequence[str], # 目标
    tokenizer: transformers.PreTrainedTokenizer, # 预训练分词器
) -> Dict:
    """Preprocess the data by tokenizing. 通过分词预处理数据。"""
    # 将源和目标拼接 
    examples = [s + t for s, t in zip(sources, targets)] 
    # 分词
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)] # 分词函数
    # [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)] 是一个列表推导式，它对 examples 和 sources 中的每个元素执行 _tokenize_fn 函数，并将结果收集到一个列表中。

    # 返回分词后的数据
    input_ids = examples_tokenized["input_ids"]
    # 复制输入标记符
    labels = copy.deepcopy(input_ids)
    # 遍历标签和源长度
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
        # 将标签中的前source_len个标记符替换为忽略索引
        # 这种技术通常用于处理变长序列。在深度学习中，由于模型通常需要固定长度的输入，因此我们经常需要对序列进行填充（padding）以使它们的长度相同。然而，这些填充的元素并不包含有用的信息，因此在计算损失函数或评估模型性能时，我们希望忽略它们。
        # 这段代码将 label 中的前 source_len 个元素设置为 IGNORE_INDEX。这可能是因为这部分元素对应于源序列（source sequence）。在训练时，我们希望模型能够预测目标序列（target sequence），而不是源序列。因此，我们将源序列对应的元素设置为 IGNORE_INDEX，这样模型在计算损失函数时就会忽略这部分元素。
    # 返回输入标记符和标签
    return dict(input_ids=input_ids, labels=labels)

# 监督数据集
class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning. 用于监督微调的数据集。"""

    # 初始化函数
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        # 调用父类初始化函数
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data... 加载数据...")
        # 加载数据集
        list_data_dict = list(utils.load_datasets(data_path))

        logging.warning("Formatting inputs... 格式化输入...")
        # 提示输入和无提示输入
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        # 获取输入和输出
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ] # 根据提示输入和无提示输入格式化输入
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict] # 根据输出格式化输出
        # tokenizer.eos_token用于获取结束标记

        logging.warning("Tokenizing inputs... This may take some time... 分词输入... 这可能需要一些时间...")
        # 预处理
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"] # 输入标记符
        self.labels = data_dict["labels"] # 标签

    # 获取长度
    def __len__(self):
        return len(self.input_ids)

    # 获取项目
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])
        # 返回输入标记符和标签


# 定义监督数据集的数据收集器
@dataclass
class DataCollatorForSupervisedDataset(object): 
    """Collate examples for supervised fine-tuning. 为监督微调收集示例。"""
    # 分词器
    tokenizer: transformers.PreTrainedTokenizer
    
    # 调用函数
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # 获取输入标记符和标签
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        # 这段代码是在从 instances 中提取 input_ids 和 labels，并将它们组合成一个元组。
        # for key in ("input_ids", "labels") 是一个生成器表达式，它遍历元组 ("input_ids", "labels") 中的每个 key，并对每个 key 执行上述的列表推导式。
        # 填充标记符
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        ) # pad_sequence()方法用于填充序列
        # torch.nn.utils.rnn是PyTorch的一个模块，用于处理序列数据
        # 填充标签
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        # 返回输入标记符、标签和注意力掩码
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

# 定义监督数据模块
def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning. 为监督微调制作数据集和收集器。"""
    # 训练数据集
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path)
    # 数据收集器
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    # DataCollatorForSupervisedDataset用于收集监督微调的数据集
    # 收集器用于将数据集中的数据收集到一起
    # 返回数据集和数据收集器
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

# 训练
def train():
    # 解析器
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments)) # HfArgumentParser用于解析命令行参数
    # 解析参数
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # parse_args_into_dataclasses()方法用于解析参数

    # 模型
    model = transformers.AutoModelForCausalLM.from_pretrained( 
        # AutoModelForCausalLM用于自动加载模型
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

    # 分词器
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        # AutoTokenizer用于自动加载分词器
        model_args.model_name_or_path, # 模型名称或路径
        cache_dir=training_args.cache_dir, # 缓存目录
        model_max_length=training_args.model_max_length, # 模型最大长度
        padding_side="right", # 填充方向
        use_fast=False, # 是否使用快速分词
    )
    # 特殊标记字典
    special_tokens_dict = dict()
    # 如果填充标记为空
    if tokenizer.pad_token is None: # tokenizer.pad_token用于获取填充标记
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    # 如果结束标记为空
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    # 如果开始标记为空
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    # 如果未知标记为空
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    # 智能分词器和嵌入调整
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    # 数据模块
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    # 训练器
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module) # Trainer的参数包括模型、分词器、训练参数和数据模块
    # 训练
    trainer.train()
    # 保存状态
    trainer.save_state() # 在训练过程中保存模型的状态，以便在训练过程中出现问题时可以恢复模型的状态
    # 保存模型
    trainer.save_model(output_dir=training_args.output_dir)

# 主函数
if __name__ == "__main__":
    train()
