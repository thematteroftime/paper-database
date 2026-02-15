# app.py
import gradio as gr
from front import demo  # 直接复用 front.py 中构建好的 Blocks 对象

if __name__ == "__main__":
    # 按魔搭要求绑定 0.0.0.0:7860
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
    )