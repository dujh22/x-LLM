import torch # pytorch
import os # 用于文件路径操作

# 显式地设置CUDA设备
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 仅使第一个GPU对当前进程可见
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from dataclasses import dataclass, field # dataclass是一个装饰器，用于简化类的定义; field是一个函数，用于定义类的属性
from tqdm import tqdm # 用于显示进度条
from typing import Optional, List # 用于类型提示
from datasets import load_dataset # 用于加载数据集

import json # 用于读写json文件
import transformers # huggingface的transformers库
from transformers import GenerationConfig # 用于生成文本

import re # 用于正则表达式
import copy # 用于深拷贝

# 从train.py中导入一些函数和变量
from train import smart_tokenizer_and_embedding_resize, \
	DEFAULT_PAD_TOKEN, DEFAULT_EOS_TOKEN, DEFAULT_BOS_TOKEN, DEFAULT_UNK_TOKEN, \
	PROMPT_DICT, \
    DataArguments
# smart_tokenizer_and_embedding_resize是一个函数，用于调整tokenizer和embedding的大小
# DEFAULT_PAD_TOKEN, DEFAULT_EOS_TOKEN, DEFAULT_BOS_TOKEN, DEFAULT_UNK_TOKEN是一些默认的特殊token
# PROMPT_DICT是一个字典，用于存储prompt的模板
# DataArguments是一个类，用于存储数据相关的参数

import train # 从train.py中导入所有内容



@dataclass
class ModelArguments(train.ModelArguments): # 继承自train.ModelArguments
    # 由于ModelArguments中的参数都是可选的，所以这里不需要再定义一遍
    # 定义一个新的参数
    load_in_8bit: bool = field( 
        default=False, # 默认值
        metadata={"help": "Load the model in 8-bit mode. 以8位模式加载模型。"}, # 参数的帮助信息
    )
    # 定义一个新的参数
    torch_dtype: torch.dtype = field(
        default=torch.bfloat16, # 默认值为torch.bfloat16
        metadata={"help": "The dtype to use for inference. 用于推理的数据类型。"},
    )


@dataclass
class GeneratingArguments: # 定义一个新的类
    # 定义一些参数
    # batch_size表示每次推理的样本数量
    batch_size: int = field(default=8) 
    # output_file表示输出文件的路径
    output_file: str = field(default=None, metadata={"help": "Path to the output."})
    # temperature表示温度参数
    temperature: float = field(default=0.7)
    # do_sample表示是否使用采样
    do_sample: bool = field(default=False)
    # top_p表示top-p采样的p值
    top_p: float = field(default=0.75)
    # top_k表示top-k采样的k值
    top_k: float = field(default=40)
    # num_beams表示beam search的beam大小，beam search表示在生成文本时，每次生成多个候选，然后选择其中最好的
    num_beams: int = field(default=1)
    # max_new_tokens表示生成文本的最大长度
    max_new_tokens: int = field(default=512)
    # template表示prompt的模板
    template: str = field(default="alpaca")
    # labels表示用于计算perplexity的标签
    labels: Optional[List[str]] = field(default=None)
    # transcot表示是否进行翻译
    transcot: bool = field(default=False)
    # transcot_skip_example表示是否跳过示例
    transcot_skip_example: bool = field(default=False)
    # evaluate表示评估方式
    evaluate: str = field(default="generate")

# 定义一个函数，用于推理
def inference():
    # 创建一个解析器
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, GeneratingArguments)) # 传入三个参数类: ModelArguments, DataArguments, GeneratingArguments分别表示模型参数、数据参数和推理参数
    # 解析参数
    model_args, data_args, generating_args = parser.parse_args_into_dataclasses()

    # 加载模型
    # transformers.AutoModelForCausalLM.from_pretrained从预训练的模型中加载一个因果语言模型
    model = transformers.AutoModelForCausalLM.from_pretrained( 
        model_args.model_name_or_path, # 模型的名称或路径
        load_in_8bit=model_args.load_in_8bit, # 是否以8位模式加载模型
        torch_dtype=model_args.torch_dtype, # 推理时使用的数据类型
        # device_map="auto",  # 设备映射
    ).to(device)  # 将模型移动到指定的设备
    # 将模型设置为评估模式
    model.eval()

    # 如果有多个GPU，则使用accelerate库
    if torch.cuda.device_count() > 1:
        # 从accelerate库中导入load_checkpoint_and_dispatch函数
        from accelerate import load_checkpoint_and_dispatch
        # 使用load_checkpoint_and_dispatch函数加载模型
        load_checkpoint_and_dispatch(
            model, # 模型
            model_args.model_name_or_path, # 模型的名称或路径
            device_map={0: "cuda:0"}, # 设备映射,原始值是"auto"
            offload_state_dict=True, # 是否卸载状态字典
            no_split_module_classes=["LlamaDecoderLayer"], # 不分割的模块类
        )
    
    # 加载tokenizer
    # 从预训练的模型中加载一个tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained( 
        model_args.model_name_or_path 
    )

    # 如果tokenizer中没有特殊token，则添加默认的特殊token
    special_tokens_dict = dict()
    if tokenizer.pad_token is None: # 如果tokenizer中没有pad_token
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN # 添加默认的pad_token
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    # 调整tokenizer和embedding的大小
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=special_tokens_dict, # 特殊token的字典
            tokenizer=tokenizer, # tokenizer
            model=model, # 模型
        )
    # 分词器的填充方向为左侧填充
    tokenizer.padding_side = "left"

    # 加载数据集
    data_path_base, data_path_name = data_args.data_path.rsplit(os.path.sep, maxsplit=1) # 将数据集路径分割为基础路径和数据集名称
    dataset_name, dataset_config = data_path_name.split("_", maxsplit=1) # 将数据集名称分割为数据集名称和配置
    test_dataset = load_dataset(os.path.join(data_path_base, dataset_name), config=dataset_config, split="test") # 加载测试集

    # 定义一个函数，用于生成prompt
    def generate_prompt(instruction, input=None, template="alpaca"):
        # 根据模板生成prompt
        if template == "alpaca":
            if input:
                return PROMPT_DICT["prompt_input"].format(instruction=instruction, input=input)
            else:
                return PROMPT_DICT["prompt_no_input"].format(instruction=instruction)
        elif template == "raw":
            if input:
                return f"{instruction}\n\n{input}"
            else:
                return f"{instruction}"
        else:
            raise NotImplementedError 
            # 如果模板不是alpaca或raw，则抛出NotImplementedError异常
        
    # 定义一个函数，用于通过生成文本进行评估
    def evaluate_by_generate(
        dataset,
        template,
        generation_config
    ):
        # 生成文本
        prompt = [generate_prompt(ins, inp, template) for ins, inp in zip(dataset["instruction"], dataset["input"])]
        # 将prompt转换为模型输入
        inputs = tokenizer(prompt, padding=True, return_tensors="pt").to(device)
        # 不进行梯度计算
        with torch.no_grad(): 
            # 生成文本
            generation_output = model.generate(
                input_ids=inputs["input_ids"], # 输入的token id
                attention_mask=inputs["attention_mask"],  # 注意力mask
                generation_config=generation_config, # 生成文本的配置
                return_dict_in_generate=True, # 返回生成文本的字典
                output_scores=True, # 输出分数
            )
        # 将生成的文本转换为字符串
        output = tokenizer.batch_decode(generation_output.sequences, skip_special_tokens=True) 
        # 返回结果
        return dataset | {"prediction": [o[len(p):].strip() for p, o in zip(prompt, output)]}
    
    # 定义一个函数，用于通过perplexity进行评估，preplexity表示模型对给定文本的预测困难程度
    def evaluate_by_perplexity(
        dataset,
        template,
        labels
    ):
        label_perplexity = [] # 用于存储每个标签的perplexity
        for label in labels:
            prompt = [generate_prompt(ins, inp, template) + label for ins, inp in zip(dataset["instruction"], dataset["input"])]
            inputs = tokenizer(prompt, padding=True, return_tensors="pt").to(device)
            # 不进行梯度计算
            with torch.no_grad():
                out_logits = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"]
                ).logits # 模型的输出，logits表示模型的输出
            shift_logits = out_logits[..., :-1, :].contiguous() # 将logits向左移动一位
            shift_targets = inputs["input_ids"][..., 1:].contiguous() # 将target向左移动一位
            shift_attention_mask_batch = inputs["attention_mask"][..., 1:].contiguous() # 将attention mask向左移动一位
            # 计算perplexity
            perplexity = torch.exp(
                (torch.nn.CrossEntropyLoss(reduction="none")(shift_logits.transpose(1, 2), shift_targets) * shift_attention_mask_batch).sum(1)
                / shift_attention_mask_batch.sum(1)
            ) # 交叉熵损失
            # 首先，代码定义了一个变量perplexity，它是一个PyTorch张量。
            # 接下来，它计算了交叉熵损失（cross-entropy loss），但是没有对损失进行reduce操作。这意味着，它会返回每个样本的交叉熵损失。
            # 然后，它对计算出的交叉熵损失乘以注意力掩码（attention mask），以确保在计算困惑度时不会考虑不必要的token。
            # 接下来，它对乘后的结果进行求和操作，横跨序列长度（sequence length）维度。这将使我们得到一个包含每个样本的困惑度值的张量。
            # 最后，它计算了每个样本的困惑度值的平均值，并将其作为最终的困惑度输出。
            label_perplexity.append(perplexity) # 将计算出的困惑度添加到 label_perplexity 列表中
        # 预测标签
        prediction = [labels[l] for l in torch.stack(label_perplexity).argmin(dim=0).detach().cpu()] # 首先将 label_perplexity 列表转换为一个张量，然后找到困惑度最小的样本的索引，然后使用这些索引从 labels 中选择对应的预测标签。
        return dataset | {"prediction": prediction} # 将预测结果添加到 dataset 中，并返回更新后的 dataset
    
    # 配置模型的生成过程
    generation_config = GenerationConfig(
        temperature=generating_args.temperature,  # 生成文本的温度，控制生成文本的多样性
        do_sample=generating_args.do_sample,  # 是否使用采样方式生成文本
        top_p=generating_args.top_p,  # 采样时的概率阈值，控制生成文本的多样性
        top_k=generating_args.top_k,  # 采样时的top-k值，控制生成文本的多样性
        num_beams=max(2, generating_args.num_beams) if generating_args.labels else generating_args.num_beams,  # beam search的数量，控制生成文本的准确性和多样性
        max_new_tokens=generating_args.max_new_tokens,  # 生成文本的最大长度
        force_word_ids=[tokenizer(generating_args.labels, add_special_tokens=False)["input_ids"]] if generating_args.labels else None  # 强制生成指定的单词
    )
    
    if generating_args.transcot:
        # 创建一个翻译缓存字典
        translation_cache = dict()
        # 配置翻译生成过程
        translation_generation_config = GenerationConfig(
            temperature=generating_args.temperature,
            do_sample=generating_args.do_sample,
            top_p=generating_args.top_p,
            top_k=generating_args.top_k,
            num_beams=generating_args.num_beams,
            max_new_tokens=512,
        )
    
    with open(generating_args.output_file, "w") as output_file:
        # 逐批次处理测试数据集
        for i in tqdm(range(0, len(test_dataset), generating_args.batch_size)):
            # 获取当前批次的数据
            d = test_dataset[i:i + generating_args.batch_size]

            
            ## ? translate input
            # 检查是否需要进行翻译
            if generating_args.transcot:
                # 复制输入数据作为原始输入
                d["original_input"] = copy.deepcopy(d["input"])
                # 定义分隔符
                _DELIM = "\n"
                _EXAMPLE_DELIM = "\n\n"
                # 创建翻译数据集
                trans_dataset = {
                    "input": [],
                    "sample_id": [],
                    "line_id": [],
                    "trans_input": [],
                }
                ### build translation dataset
                # 创建一个翻译数据集
                for sample_id, sample in enumerate(d["input"]):
                    # 遍历每个样本
                    if generating_args.transcot_skip_example:
                        # 如果设置了transcot_skip_example为True
                        lines_to_translate = sample.split(_EXAMPLE_DELIM)[-1].split(_DELIM)
                        # 将样本按照_EXAMPLE_DELIM分割，取最后一个部分，再按照_DELIM分割为多行
                    else:
                        # 如果设置了transcot_skip_example为False
                        lines_to_translate = sample.split(_DELIM)
                        # 将样本按照_DELIM分割为多行
                    for line_id, line in enumerate(lines_to_translate):
                        # 遍历每行文本
                        if re.match(r"[A-D]\. ", line) is not None:
                            # 如果行以字母和句点开头，例如"A. "
                            line = line[3:]
                            # 去掉前三个字符，即字母和句点
                        if line.strip() and line not in translation_cache.keys() and not line.startswith("Answer:"):
                            # 如果行不为空且不在翻译缓存中且不以"Answer:"开头
                            trans_dataset["input"].append(line)
                            # 将行添加到翻译数据集的input列表中
                            trans_dataset["sample_id"].append(sample_id)
                            # 将样本ID添加到翻译数据集的sample_id列表中
                            trans_dataset["line_id"].append(line_id)
                            # 将行ID添加到翻译数据集的line_id列表中
                ### run translation
                # 逐批次处理翻译数据集
                for i in range(0, len(trans_dataset["input"]), generating_args.batch_size):
                    # 获取当前批次的数据
                    td = trans_dataset["input"][i:i + generating_args.batch_size]
                    # 进行翻译生成
                    trans_output = evaluate_by_generate({
                            "input": td,
                            "instruction": ["Translate the following sentences to English."] * len(td),
                        }, 
                        template="alpaca",
                        generation_config=translation_generation_config
                    )
                    # 将翻译结果添加到翻译数据集的trans_input列表中
                    trans_dataset["trans_input"] += trans_output["prediction"]
                    # 更新翻译缓存
                    for inp, pre in zip(trans_output["input"], trans_output["prediction"]):
                        translation_cache[inp] = pre
                ### change input
                # 遍历原始输入数据中的每个样本
                d["input"] = [
                    # 将每个样本按照指定的分隔符重新组合
                    _DELIM.join([
                        # 对于每行文本进行处理
                        # 如果行不以字母和句点开头，直接使用翻译缓存中的翻译结果（如果存在），否则保持原样
                        (translation_cache[line] if line in translation_cache.keys() else line)
                        # 如果行以字母和句点开头，将前三个字符保持不变，后面的文本使用翻译缓存中的翻译结果（如果存在），否则保持原样
                        if re.match(r"[A-D]\. ", line) is None
                        else (line[:3] + translation_cache[line[3:]] if line[3:] in translation_cache.keys() else line)
                        # 遍历样本中的每行文本
                        for line in sample.split(_DELIM)
                    ])
                    # 将处理后的样本添加到新的输入数据中
                    for sample in d["original_input"]
                ]
            ## ? translate input
            ## ? 翻译输入

            # 检查评估方法是否为"generate"
            if generating_args.evaluate == "generate":
                # 使用给定的参数调用evaluate_by_generate函数
                output = evaluate_by_generate(d, template=generating_args.template, generation_config=generation_config)
            # 检查评估方法是否为"perplexity"
            elif generating_args.evaluate == "perplexity":
                # 检查是否提供了标签
                assert generating_args.labels, "evaluate with perplexity requires labels"
                # 使用给定的参数调用evaluate_by_perplexity函数
                output = evaluate_by_perplexity(d, template=generating_args.template, labels=generating_args.labels)
            
            # 将输出写入输出文件
            output_file.writelines(
                json.dumps(sample, ensure_ascii=False) + "\n" for sample in [dict(zip(output.keys(),t)) for t in zip(*output.values())]
            )
            output_file.flush()
    
if __name__ == "__main__":
    inference()
    