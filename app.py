# 导入必要的库
import torch  # torch是一个开源的机器学习库，用于深度学习和张量计算
import transformers  # transformers是一个开源的自然语言处理库，提供大量预训练模型
import gradio as gr  # gradio是一个库，用于快速创建机器学习和数据科学的Web UI
import sys  # sys模块提供了一系列有关Python运行环境的变量和函数

# 从train.py中导入特定的函数和变量
from train import smart_tokenizer_and_embedding_resize, \
    DEFAULT_PAD_TOKEN, DEFAULT_EOS_TOKEN, DEFAULT_BOS_TOKEN, DEFAULT_UNK_TOKEN, \
    PROMPT_DICT, \
    DataArguments
# 这些导入项包括用于调整tokenizer和embedding大小的函数、默认的特殊token、用于存储prompt模板的字典、以及一个用于存储数据参数的类

# 获取命令行参数中指定的基础模型名称，设置默认值为model/llama-7b-hf.alpaca_en+alpaca_zh+translation_ncwm_en-zh.finetune
BASE_MODEL = "model/llama-7b-hf.alpaca_en+alpaca_zh+translation_ncwm_en-zh.finetune"
if len(sys.argv) > 1:
    BASE_MODEL = sys.argv[1]  # 第一个命令行参数被视为基础模型的名称


# 根据系统是否支持CUDA决定使用的设备
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# 尝试检查系统是否支持MPS后端，MPS是用于Mac系统上的GPU加速
try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass  # 如果检查时发生异常，则忽略该异常

# 根据设备类型加载模型
if device == "cuda":
    model = transformers.AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        load_in_8bit=False,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
elif device == "mps":
    model = transformers.AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map={"": device},
        torch_dtype=torch.bfloat16,
    )
else:
    model = transformers.AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, device_map={"": device}, low_cpu_mem_usage=True
    )

# 加载tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(
    BASE_MODEL,
    use_fast=False,
)

# 初始化一个空字典用于存储特殊token
special_tokens_dict = dict()
# 检查并设置tokenizer的特殊token
if tokenizer.pad_token is None:
    special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
if tokenizer.eos_token is None:
    special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
if tokenizer.bos_token is None:
    special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
if tokenizer.unk_token is None:
    special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

# 如果需要，调整tokenizer和模型的embedding
if tokenizer.pad_token is None:
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )
# 设置tokenizer的padding方向
tokenizer.padding_side = "left"

# 定义一个函数用于生成prompt
def generate_prompt(instruction, input=None, template="alpaca"):
    # 根据模板类型生成不同的prompt
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

# 如果使用的不是CPU设备，将模型转为半精度以节省内存
if device != "cpu":
    model.half()
model.eval()  # 将模型设置为评估模式
if torch.__version__ >= "2":
    model = torch.compile(model)  # 如果PyTorch版本大于等于2，对模型进行编译以优化性能

# 定义评估函数
def evaluate(
    instruction,  # 指令
    input=None,  # 输入
    template="alpaca", # 使用的模板类型
    temperature=0.1, # 控制生成文本的随机性的温度参数
    top_p=0.75, # 在生成文本时，保留累积概率为top_p的最可能的词
    top_k=40, # 在生成文本时，保留概率最高的top_k个词
    num_beams=4, # beam搜索的数量，用于生成更好的文本输出
    max_new_tokens=128, # 生成文本的最大长度
    **kwargs, # 其他配置参数
):
    # 生成prompt
    prompt = generate_prompt(instruction, input, template)
    # 对prompt进行tokenize处理
    inputs = tokenizer(prompt, return_tensors="pt")
    # 将tokenized的输入转移到指定设备
    input_ids = inputs["input_ids"].to(device)
    # 设置生成文本的配置参数
    generation_config = transformers.GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        **kwargs,
    )
    with torch.no_grad():  # 禁用梯度计算
        # 使用模型生成文本
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    # 获取生成的序列
    s = generation_output.sequences[0]
    # 将生成的序列解码成文本
    output = tokenizer.decode(s, skip_special_tokens=True)
    # 去除prompt部分，只返回生成的文本内容
    return output[len(prompt):].strip()

# 创建一个gradio界面
g = gr.Interface(
    # 设置界面调用的函数
    fn=evaluate,
    # 设置界面的输入组件
    inputs=[
        gr.components.Textbox(
            lines=2, label="Instruction", placeholder="Tell me about alpacas."
        ),
        gr.components.Textbox(lines=2, label="Input", placeholder="none"),
        gr.components.Dropdown(["raw", "alpaca"], value="alpaca", label="Template Format"),
        gr.components.Slider(minimum=0, maximum=1, value=0.7, label="Temperature"),
        gr.components.Slider(minimum=0, maximum=1, value=0.75, label="Top p"),
        gr.components.Slider(minimum=0, maximum=100, step=1, value=40, label="Top k"),
        gr.components.Slider(minimum=1, maximum=4, step=1, value=1, label="Beams"),
        gr.components.Slider(
            minimum=1, maximum=512, step=1, value=512, label="Max tokens"
        ),
    ],
    # 设置界面的输出组件
    outputs=[
        gr.inputs.Textbox(
            lines=5,
            label="Output",
        )
    ],
    # 界面标题
    title="🦙Alpaca Web Demo",
    # 界面描述
    description="Made with 💖",
    # 提供的反馈选项
    flagging_options=["👍🏼", "👎🏼"]
)
# 设置队列并发数
g.queue(concurrency_count=1)
# 启动界面
g.launch(server_name='0.0.0.0', server_port=8086)

# Old testing code follows.

"""
if __name__ == "__main__":
    # testing code for readme
    for instruction in [
        "Tell me about alpacas.",
        "Tell me about the president of Mexico in 2019.",
        "Tell me about the king of France in 2019.",
        "List all Canadian provinces in alphabetical order.",
        "Write a Python program that prints the first 10 Fibonacci numbers.",
        "Write a program that prints the numbers from 1 to 100. But for multiples of three print 'Fizz' instead of the number and for the multiples of five print 'Buzz'. For numbers which are multiples of both three and five print 'FizzBuzz'.",
        "Tell me five words that rhyme with 'shock'.",
        "Translate the sentence 'I have no mouth but I must scream' into Spanish.",
        "Count up from 1 to 500.",
    ]:
        print("Instruction:", instruction)
        print("Response:", evaluate(instruction))
        print()
"""
