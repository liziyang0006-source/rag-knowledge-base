# 基础镜像：python 3.11 slim（体积小、稳定，依赖兼容性好）
FROM python:3.11-slim

# 容器最佳实践：不生成 .pyc、日志实时输出
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# pip 用清华镜像源，国内构建下载依赖快得多
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn

WORKDIR /app

# 先单独复制 requirements.txt 安装依赖：
# 依赖层和代码层分离，改动代码重新 build 时不会重装依赖（利用层缓存）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码（.dockerignore 已排除 .venv/.env/缓存/本地数据等）
COPY . .

# FastAPI 服务端口
EXPOSE 8000

# 启动命令：uvicorn 直接拉起 api.main:app
# api/main.py 里的 sys.path.insert 会把 /app 加进路径，from app.xxx import 正常工作
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]