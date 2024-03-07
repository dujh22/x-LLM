# coding=utf-8

# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# 版权 2020 TensorFlow 数据集作者和 HuggingFace 数据集作者。
#
# 根据 Apache 许可证 2.0 版（“许可证”）获得许可;
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下位置获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则软件
# 根据“原样”分发，不附带任何明示或暗示的担保或条件。
# 请参阅许可证了解特定语言的权限和限制。

# Lint as: python3

# dummy dataset uses load datasets for alias 
# 虚拟数据集使用别名的加载数据集

import os
import json
import datasets
logger = datasets.logging.get_logger(__name__) # 创建一个记录器

# 用于 Alias 的 BuilderConfig
class AliasConfig(datasets.BuilderConfig):

    """BuilderConfig for Alias 用于 Alias 的 BuilderConfig"""

    def __init__(self, config: str, **kwargs):
        """
        Args:
            lang: string, language for the input text（字符串，输入文本的语言）
            **kwargs: keyword arguments forwarded to super.（转发给 super 的关键字参数）
        """
        super(AliasConfig, self).__init__(**kwargs) # 调用父类构造函数
        self.alias = config

# 定义一个别名的数据集
class Alias(datasets.GeneratorBasedBuilder):
    """This is an adapter for loading raw text parallel corpus.
        这是一个用于加载原始文本并行语料库的适配器。"""
    VERSION = datasets.Version("1.0.0")
    BUILDER_CONFIG_CLASS = AliasConfig

    def _info(self):
        return datasets.DatasetInfo(
            description="", # 描述为空
            # 定义数据集的特征
            features=datasets.Features(
                {
                    "id": datasets.Value("string"), # id 是字符串
                    "instruction": datasets.Value("string"), # 指令是字符串
                    "input": datasets.Value("string"), # 输入是字符串
                    "output": datasets.Value("string"), # 输出是字符串
                    "from": datasets.Value("string"), # 来源是字符串
                }
            ),
            homepage="",  # 主页为空
            citation="",  # 引用为空
        )

    # 定义数据集的下载和处理逻辑
    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": ""}), # 返回训练分割生成器
        ]

    # 定义数据集的生成逻辑
    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form. 该函数以原始（文本）形式返回示例。"""
        logger.info("[alias] generating examples from = %s", filepath) # 记录生成示例的消息
        
        alias = json.load(open(os.path.join(self.base_path, "alias.json")))  # 从 JSON 文件中加载别名
        full_dataset = alias[self.config.alias] # 从别名获取完整数据集
        
        import sys
        sys.path.append(os.path.join(os.path.pardir, os.path.pardir)) # 将父目录添加到 sys 路径中

        from utils import load_datasets
        
        # 生成数据集
        for _id, d in enumerate(load_datasets(os.path.join(self.base_path, os.path.pardir, full_dataset))):
            # 生成示例 ID 和示例数据
            yield _id, d # 返回数据集
