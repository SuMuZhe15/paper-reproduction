
import os
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv


# 1. 读取项目根目录下的 .env
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=True)

# 2. 初始化 DeepSeek 客户端
API_KEY = os.getenv("DEEPSEEK_API_KEY")

if not API_KEY:
    raise ValueError(
        "没有读取到 DEEPSEEK_API_KEY，请检查项目根目录下的 .env"
    )

client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.deepseek.com",
)

MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")


# 3. 统一的模型调用接口
def ask_llm(prompt: str, system_prompt: str = "You are a helpful assistant.") -> str:
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    )

    return response.choices[0].message.content or ""


# 4. 单独运行本文件时，测试 API
if __name__ == "__main__":
    answer = ask_llm("请只回答：ReAct API 测试成功！")
    print(answer)

