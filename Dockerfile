FROM python:3.10-slim

WORKDIR /app

# 复制项目文件
COPY ./ /app

# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt

# 暴露魔搭要求的端口
EXPOSE 7860

# 启动 Gradio 应用
CMD ["python", "-u", "app.py"]