import os
import sys
from dotenv import load_dotenv
from langchain_dashscope import ChatDashScope
import traceback

print(f"--- 正在使用 Python 版本: {sys.version_info.major}.{sys.version_info.minor} ---")

# 1. 加载 .env 文件
print("--- 正在加载 .env 文件... ---")
load_dotenv(override=True)

# 2. 从 .env 读取密钥
API_KEY = os.getenv("DASHSCOPE_API_KEY")

# 3. 打印-调试
if not API_KEY:
    print("!!! 错误：在 .env 中未找到 DASHSCOPE_API_KEY。!!!")
    print("请确保 .env 文件和本脚本在同一目录下。")
    exit()
else:
    # 打印部分密钥以确认，这很安全
    print(f"成功加载 API Key (开头/结尾): {API_KEY[:4]}...{API_KEY[-4:]}")

try:
    # 4. 实例化
    print("--- 正在尝试初始化 ChatDashScope... ---")
    model_test = ChatDashScope(api_key=API_KEY)
    
    if model_test.client is None:
        print("!!! 错误：ChatDashScope 初始化后 self.client 依然是 None。!!!")
    else:
        print("+++ ChatDashScope 初始化成功 (self.client 已设置)。 +++")

    # 5. 尝试调用
    question = "你好，请你做个自我介绍"
    print(f"--- 正在尝试调用 .invoke()... 问题: {question} ---")
    result = model_test.invoke(question)
    
    # 6. 打印结果
    print("\n--- [调用成功] ---")
    print(result.content)
    print("------------------")
    
except Exception as e:
    print(f"\n--- [!!! 发生错误 !!!] ---")
    print(f"错误类型: {type(e)}")
    print(f"错误信息: {e}")
    print("--- [详细堆栈跟踪] ---")
    traceback.print_exc()
    print("-----------------------")