import os
import sys
import traceback
from dotenv import load_dotenv

# 1. 必须先加载 .env
print("--- 正在加载 .env 文件... ---")
load_dotenv(override=True)

try:
    # 2. 导入 dashscope 库
    import dashscope
    print("+++ 'dashscope' 库导入成功 +++")
    
except ImportError as e:
    print("!!! 致命错误：'dashscope' 库未安装或导入失败。!!!")
    print(f"请运行: pip install dashscope")
    print(f"错误: {e}")
    exit()

# 3. 从环境变量设置 API Key
API_KEY = os.getenv("DASHSCOPE_API_KEY")

if not API_KEY:
    print("!!! 致命错误：未能在 .env 中找到 DASHSCOPE_API_KEY。!!!")
    exit()
else:
    print(f"成功加载 API Key (开头/结尾): {API_KEY[:4]}...{API_KEY[-4:]}")

# 4. 这是 dashscope 库的标准用法
dashscope.api_key = API_KEY

try:
    print("--- 正在尝试调用 'dashscope.Generation.call()' ... ---")
    
    # 5. 发起请求
    response = dashscope.Generation.call(
        model='qwen-turbo',
        prompt='你好，请你做个自我介绍'
    )
    
    # 6. 检查结果
    if response.status_code == 200:
        print("\n--- [调用成功] ---")
        print(response.output.text)
        print("------------------")
    else:
        print(f"\n--- [!!! 调用失败 !!!] ---")
        print(f"HTTP 状态码: {response.status_code}")
        print(f"错误代码: {response.code}")
        print(f"错误信息: {response.message}")
        print("-----------------------")

except Exception as e:
    print(f"\n--- [!!! 发生严重异常 !!!] ---")
    print(f"错误类型: {type(e)}")
    print(f"错误信息: {e}")
    print("--- [详细堆栈跟踪] ---")
    traceback.print_exc()
    print("-----------------------")