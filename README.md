# PlasmaRAG - 复杂等离子体物理论文智能分析系统

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Gradio](https://img.shields.io/badge/Gradio-6.5.1-green.svg)](https://gradio.app/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

一个基于RAG（检索增强生成）技术的复杂等离子体物理论文智能分析系统，能够自动提取论文中的物理参数、力场模型、实验现象等关键信息，并为用户提供智能化的模拟参数推荐。

## ✨ 功能特性

### 📄 论文智能解析
- **多阶段提取流水线**：使用 `qwen-long` 进行深度物理理解，`qwen-plus` 进行严格JSON格式化
- **结构化信息提取**：自动提取论文标题、期刊、年份、创新点、物理背景、实验现象等
- **物理参数识别**：智能识别并分类物理参数（几何参数、电学参数、无量纲参数等）
- **力场模型提取**：自动提取微粒间相互作用势（Pair Potentials）的数学公式和物理意义

### 🖼️ 图表智能分析
- **自动图表提取**：从PDF中提取页面图像并保存
- **视觉模型标注**：使用 `qwen-vl-max` 视觉大模型为图表生成物理语义说明
- **参数关联**：自动识别图表与物理参数的关联关系

### 🔍 向量检索与知识库
- **双索引系统**：使用FAISS构建论文和力场的独立向量索引
- **SQLite元数据存储**：持久化存储论文和力场的详细元数据
- **智能查重**：基于标题和公式哈希的自动去重机制
- **向量检索**：基于语义相似度的知识检索

### 💡 智能模拟推荐
- **参数区间推荐**：根据参考论文和用户需求，智能推荐模拟参数的合理区间
- **步长建议**：基于物理特征尺度推荐合适的参数扫描步长
- **力场模型推荐**：推荐最适合的相互作用势模型
- **现象匹配**：根据期望观察到的物理现象调整推荐策略

### 🎨 现代化Web界面
- **模块化设计**：Paper Analysis、Simulation Setup、Library 三大功能模块
- **实时进度显示**：可视化展示论文解析和入库进度
- **美观的卡片布局**：专业的物理参数和力场展示界面
- **LaTeX公式渲染**：支持数学公式的美观展示
- **响应式设计**：适配不同屏幕尺寸

## 🛠️ 技术栈

### 核心框架
- **Gradio 6.5.1**：Web界面框架
- **OpenAI SDK**：兼容DashScope API的LLM调用
- **FAISS**：高效的向量相似度搜索
- **SQLite**：轻量级关系型数据库

### AI模型
- **qwen-long**：深度物理理解与信息提取
- **qwen-plus**：JSON格式化与结构化输出
- **qwen-vl-max**：视觉理解与图表标注
- **text-embedding-v2**：文本向量化

### 数据处理
- **PyMuPDF (fitz)**：PDF解析与图像提取
- **NumPy**：数值计算
- **Pandas**：数据处理与表格展示

## 📦 安装指南

### 环境要求
- Python 3.10 或更高版本
- 8GB+ RAM（推荐）
- 网络连接（用于API调用）

### 安装步骤

1. **克隆仓库**
```bash
git clone https://github.com/yourusername/paper_web.git
cd paper_web
```

2. **创建虚拟环境（推荐）**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **安装依赖**
```bash
pip install -r requirements.txt
```

4. **配置API密钥**

创建 `.env` 文件或设置环境变量：
```bash
# Windows PowerShell
$env:DASHSCOPE_API_KEY="your-api-key-here"

# Linux/Mac
export DASHSCOPE_API_KEY="your-api-key-here"
```

或者在 `backend.py` 中直接配置（不推荐用于生产环境）。

## 🚀 使用方法

### 启动应用

```bash
python app.py
```

应用将在 `http://localhost:7860` 启动。

### 基本工作流程

1. **论文分析**
   - 点击"上传PDF"按钮或拖拽PDF文件
   - 点击"🚀 分析并入库"按钮
   - 等待解析完成（包含上传、解析、物理提取、向量化、入库等步骤）
   - 查看提取的结构化信息

2. **模拟参数推荐**
   - 切换到"Simulation Setup"模块
   - 输入期望观察到的物理现象
   - 填写或修改模拟参数表（参数名称、目标数值、单位、物理意义）
   - 点击"💡 生成对标推荐报告"
   - 查看推荐的参数区间、步长和力场模型

3. **知识库浏览**
   - 切换到"Library"模块
   - 查看已入库的所有论文
   - 点击表格中的论文行，然后点击"📖 查看选中论文"查看详细信息

## ⚙️ 配置说明

### API配置

项目使用阿里云DashScope API，需要配置以下内容：

- **DASHSCOPE_API_KEY**：API密钥（必需）
- **Base URL**：`https://dashscope.aliyuncs.com/compatible-mode/v1`（已内置）

### 数据库配置

- **SQLite数据库**：默认路径为 `plasma_knowledge.db`
- **FAISS索引**：
  - 论文索引：`faiss_papers.index`
  - 力场索引：`faiss_forces.index`

### 向量维度

- 默认向量维度：1536（text-embedding-v2模型输出维度）

## 📁 项目结构

```
paper_web/
├── app.py                 # 应用入口（魔搭部署配置）
├── backend.py             # 核心RAG系统实现
├── front.py               # Gradio前端界面
├── requirements.txt       # Python依赖列表
├── Dockerfile             # Docker容器化配置
├── README.md             # 项目说明文档
├── plasma_knowledge.db    # SQLite数据库（运行后生成）
├── faiss_papers.index     # 论文向量索引（运行后生成）
├── faiss_forces.index     # 力场向量索引（运行后生成）
├── figures/              # 提取的图表存储目录
│   └── [论文名]/
│       └── *.png
└── images/                  # 示例图片资源
    └── *.png
```

## 🔧 核心模块说明

### ComplexPlasmaRAG 类

主要的RAG系统类，提供以下核心功能：

- `extract_paper_structure(file_path)`: 提取论文结构化信息
- `extract_figures(file_path, structured_text)`: 提取并标注图表
- `update_vector_db(structured_data)`: 更新向量数据库
- `search_knowledge(query_text, top_k)`: 语义检索相关知识
- `get_simulation_recommendation(structured_paper, user_params)`: 生成模拟推荐

### 前端界面模块

- **Paper Analysis**：论文解析与展示
- **Simulation Setup**：模拟参数配置与推荐
- **Library**：知识库浏览与管理

## 🐳 Docker部署

### 构建镜像

```bash
docker build -t plasmarag:latest .
```

### 运行容器

```bash
docker run -d \
  -p 7860:7860 \
  -e DASHSCOPE_API_KEY=your-api-key \
  -v $(pwd)/plasma_knowledge.db:/app/plasma_knowledge.db \
  -v $(pwd)/faiss_papers.index:/app/faiss_papers.index \
  -v $(pwd)/faiss_forces.index:/app/faiss_forces.index \
  plasmarag:latest
```

## 📝 使用示例

### 示例1：解析论文并入库

```python
from backend import ComplexPlasmaRAG

# 初始化RAG系统
rag = ComplexPlasmaRAG(api_key="your-api-key")

# 解析PDF论文
structured_data = rag.extract_paper_structure("paper.pdf")

# 入库
rag.update_vector_db(structured_data)
```

### 示例2：检索相关知识

```python
# 语义检索
papers, forces = rag.search_knowledge("电变流变等离子体 链状结构", top_k=3)

# 查看检索结果
for paper in papers:
    print(paper['metadata']['title'])
```

### 示例3：生成模拟推荐

```python
# 用户参数
user_params = {
    "target_particle_charge": {
        "value": "1.2 * 10^4",
        "unit": "e",
        "description": "目标微粒电荷"
    },
    "expected_phenomena": "观察到链状结构形成"
}

# 生成推荐
recommendation = rag.get_simulation_recommendation(structured_data, user_params)
print(recommendation)
```

## ⚠️ 注意事项

1. **API配额**：项目依赖DashScope API，请注意API调用配额和费用
2. **PDF质量**：建议使用高质量的PDF文件，包含清晰的文本和图表
3. **数据持久化**：定期备份 `plasma_knowledge.db` 和FAISS索引文件
4. **并发安全**：系统使用文件锁机制防止并发写入冲突
5. **内存占用**：FAISS索引会占用一定内存，建议在内存充足的环境运行

## 🤝 贡献指南

欢迎贡献代码、报告问题或提出建议！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 🙏 致谢

- [Gradio](https://gradio.app/) - 优秀的Web界面框架
- [FAISS](https://github.com/facebookresearch/faiss) - 高效的向量检索库
- [DashScope](https://dashscope.aliyun.com/) - 阿里云AI服务
- [Qwen](https://github.com/QwenLM/Qwen) - 强大的大语言模型

## 📧 联系方式

如有问题或建议，请通过以下方式联系：

- 提交 [Issue](https://github.com/yourusername/paper_web/issues)
- 发送邮件至：your-email@example.com

## 🔮 未来计划

- [ ] 支持更多文件格式（Word、LaTeX等）
- [ ] 增强图表识别精度（精确裁剪图表区域）
- [ ] 支持多语言论文解析
- [ ] 添加论文对比功能
- [ ] 集成更多物理模拟工具
- [ ] 支持批量论文处理
- [ ] 添加用户认证和权限管理
- [ ] 优化向量检索性能

---

**如果这个项目对你有帮助，请给一个 ⭐ Star！**

