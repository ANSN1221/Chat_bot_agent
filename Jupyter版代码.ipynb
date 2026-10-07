import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
load_dotenv(override=True)

# 1. 基础配置
MODEL_NAME = "deepseek-v4-flash"
MAX_PAIRS_HISTORY = 10
EXIT_WORD = "quit"

# 2. 初始化模型
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")
model = init_chat_model(
    #model="deepseek-v4-flash",
    api_key=DEEPSEEK_API_KEY,
    #model_provider= "deepseek",
    model = "deepseek:deepseek-v4-flash",
    base_url=DEEPSEEK_BASE_URL,
)
# json 格式的消息
def keep_recent_messages(messages, max_pairs=3):
    """
    保留最近的 N 轮对话

    max_pairs: 保留的对话轮数（每轮 = user + assistant）
    """
    # 分离 system 和对话
    system_msgs = [m for m in messages if m.get("role") == "system"]
    conversation_msgs = [m for m in messages if m.get("role") != "system"]

    # 只保留最近的
    recent_msgs = conversation_msgs[-(max_pairs * 2):]

    # 返回：system + 最近对话
    return system_msgs + recent_msgs
# 3. 初始化消息列表
messages = [
    {
        "role":"system",
        "content":"计科专业的开发者，专业是Agent开发，也是一名耐心、友好的智能助手。我会用自然、清晰的方式回答用户问题。"
    }
]

# 4. 启动提示
print(f"✨ 请输入问题，输入 {EXIT_WORD} 结束对话\n")

# 5. 多轮对话主循环
# 轮次记录
i = 1
while True:
    print("\n", "=" * 10, f'-> 第 {i} 轮对话开始 <-', "=" * 10, "\n")
    user_input = input("🙋 请输入：")

    # 退出判断
    if user_input.lower() == EXIT_WORD:
        print("🌙 对话已结束，欢迎下次再来！")
        break

    # 追加用户消息
    messages.append({"role":"user","content":user_input})

    # 流式输出模型回复
    print("🧚ANSN：", end="", flush=True)

    reply_content = ""

    # 优化历史记忆
    memory_messages = keep_recent_messages(messages,max_pairs =MAX_PAIRS_HISTORY)
    # 控制发送给模型的消息长度
    for chunk in model.stream(memory_messages):
        if chunk.content:
            print(chunk.content, end="", flush=True)
            reply_content += chunk.content

    print("\n", "=" * 10, f'-> 第 {i} 轮对话结束 <-', "=" * 10, "\n")
    i += 1

    # 追加 AI 回复
    messages.append({"role":"assistant","content":reply_content})
