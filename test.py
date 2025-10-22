import os
import sys
import traceback

# ----------------- 关键修复 -----------------
# 必须在导入任何 dashscope 库之前加载 .env
from dotenv import load_dotenv
print("--- 正在加载 .env 文件... ---")
load_dotenv(override=True)
# ----------------------------------------------

# 现在 .env 已经加载，我们可以安全地导入它们了
from langchain_dashscope import ChatDashScope

print(f"--- 正在使用 Python 版本: {sys.version_info.major}.{sys.version_info.minor} ---")

# 2. 从 .env 读取密钥
API_KEY = os.getenv("DASHSCOPE_API_KEY")

# 3. 打印-调试
if not API_KEY:
    print("!!! 错误：在 .env 中未找到 DASHSCOPE_API_KEY。!!!")
    print("请确保 .env 文件和本脚本在同一目录下。")
    exit()
else:
    print(f"成功加载 API Key (开头/结尾): {API_KEY[:4]}...{API_KEY[-4:]}")
    
try:
    # 4. 实例化
    print("--- 正在尝试初始化 ChatDashScope... ---")
    
    # ----------------- 关键修复 -----------------
    # 不要手动传递 api_key。
    # 让库自动从我们刚刚加载的环境变量中读取。
    # 这是更健壮的标准做法。
    model_test = ChatDashScope() 
    # ----------------------------------------------

    if model_test.client is None:
         print("!!! 警告：ChatDashScope 初始化后 self.client 依然是 None。(这在某些新版本中是正常的)")
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