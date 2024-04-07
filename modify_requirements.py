import re

# 打开文件并读取内容
with open('requirements2024032502.txt', 'r') as file:
    content = file.read()

# 使用正则表达式删除所有的版本号
content = re.sub(r'==\d+\.\d+\.\d+', '', content)

# 将修改后的内容写回文件
with open('requirements2024032502.txt', 'w') as file:
    file.write(content)