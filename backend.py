import os
from pathlib import Path
from openai import OpenAI
import faiss
import numpy as np
import sqlite3
import hashlib
import json
import portalocker
import fitz  # PyMuPDF，用于从 PDF 中提取图像
import base64

# 1. 初始化客户端与配置
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY") or "sk-fd7afdef962a46d39784e8b0b8133974",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

PROJECT_ROOT = Path(__file__).resolve().parent


class ComplexPlasmaRAG:
    def __init__(self, db_path="plasma_knowledge.db",
                 paper_idx_path="faiss_papers.index",
                 force_idx_path="faiss_forces.index",
                 api_key=None):
        # 如果传入了 api_key，就更新全局的 client（或者新建一个）
        if api_key:
            global client  # 引用外部的 client 对象
            client = OpenAI(
                api_key=api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
        self.db_path = db_path
        self.paper_idx_path = paper_idx_path
        self.force_idx_path = force_idx_path
        self.dimension = 1536

        # 1. 初始化 SQLite 数据库 (用于元数据持久化和查重)
        self._init_sqlite()

        # 2. 初始化或加载 FAISS 索引
        if os.path.exists(self.paper_idx_path):
            index = faiss.read_index(self.paper_idx_path)
            if not isinstance(index, faiss.IndexIDMap):
                print("⚠️ 索引不是 IDMap，自动包装...")
                index = faiss.IndexIDMap(index)
            self.paper_index = index

            # self.paper_index = faiss.read_index(self.paper_idx_path)
            print(f"从磁盘加载论文索引，当前规模: {self.paper_index.ntotal}")
        else:
            # self.paper_index = faiss.IndexFlatL2(self.dimension)
            base_index = faiss.IndexFlatL2(self.dimension)
            self.paper_index = faiss.IndexIDMap(base_index)

        if os.path.exists(self.force_idx_path):
            index = faiss.read_index(self.force_idx_path)
            if not isinstance(index, faiss.IndexIDMap):
                print("⚠️ 索引不是 IDMap，自动包装...")
                index = faiss.IndexIDMap(index)
            self.force_index = index

            # self.force_index = faiss.read_index(self.force_idx_path)
            print(f"从磁盘加载力场索引，当前规模: {self.force_index.ntotal}")
        else:
            # self.force_index = faiss.IndexFlatL2(self.dimension)
            base_index = faiss.IndexFlatL2(self.dimension)
            self.force_index = faiss.IndexIDMap(base_index)

    def _init_sqlite(self):
        """初始化数据库表结构"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")

            cursor = conn.cursor()
            # 论文表：以标题作为唯一约束进行查重
            cursor.execute('''CREATE TABLE IF NOT EXISTS papers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT UNIQUE,
                metadata_json TEXT,
                vector_id INTEGER
            )''')
            # 力场表：以公式和背景的组合哈希作为唯一约束
            cursor.execute('''CREATE TABLE IF NOT EXISTS force_fields (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                formula_hash TEXT UNIQUE,
                force_json TEXT,
                source_paper TEXT,
                vector_id INTEGER
            )''')
            # 3. 新增：图表信息表
            # 存储每张图片的路径、标注、所属页码，以及预留的向量 ID
            cursor.execute('''CREATE TABLE IF NOT EXISTS figures (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            paper_id INTEGER,
                            image_path TEXT,
                            caption TEXT,
                            page_num INTEGER,
                            figure_vector_id INTEGER,
                            FOREIGN KEY (paper_id) REFERENCES papers (id)
                        )''')
            conn.commit()

    def extract_figures(self, file_path: str, structured_text: dict):
        """
        从 PDF 中提取图像，并尝试基于视觉模型与物理参数进行语义对标。

        当前版本（Step 2 & 3 原型）：
        - 仍按“整页截图”方式为每一页生成一张 PNG（后续可升级为精确图表裁剪）。
        - 对每一页图像，结合 structured_text['parameters']，调用视觉大模型（预留接口）推断：
          * 该图的物理含义 caption（侧重描述与模拟参数的关系）；
          * 与之最相关的物理参数列表 linked_parameters。
        - 返回的数据结构与 JSON 中 figures 字段保持一致，便于前端和数据库直接使用。
        """
        figures = []
        try:
            pdf_path = Path(file_path)
            if not pdf_path.exists():
                return figures

            # 物理参数列表（用于对标）
            params = structured_text.get("parameters", []) or []
            param_summaries = []
            for p in params:
                name = p.get("name", "")
                symbol = p.get("symbol", "")
                unit = p.get("unit", "")
                meaning = p.get("meaning", "")
                line = f"名称: {name}, 符号: {symbol}, 单位: {unit}, 含义: {meaning}"
                param_summaries.append(line)
            param_summary_text = "\n".join(param_summaries) if param_summaries else "（当前未提取到任何参数）"

            # 输出目录：<项目根>/figures/<pdf_stem>/
            base_dir = PROJECT_ROOT / "figures" / pdf_path.stem
            base_dir.mkdir(parents=True, exist_ok=True)
            print(f"[extract_figures] pdf_path={pdf_path}, base_dir={base_dir}")

            doc = fitz.open(str(pdf_path))
            max_pages_for_figures = 6  # 限制处理页数，避免过多调用 VLM

            for page_index in range(len(doc)):
                if page_index >= max_pages_for_figures:
                    break
                page = doc[page_index]
                # 先使用整页截图作为粗粒度图表（后续可演进为图表区域裁剪）
                pix = page.get_pixmap(dpi=160)
                img_name = f"{pdf_path.stem}_p{page_index + 1}.png"
                img_path = base_dir / img_name
                pix.save(str(img_path))

                abs_img_path = img_path.resolve()
                try:
                    # 对数据库与前端都存储为【相对项目根目录】的路径
                    rel_img_path = abs_img_path.relative_to(PROJECT_ROOT)
                    image_path_str = rel_img_path.as_posix()
                except ValueError:
                    # 理论上不会走到这里（base_dir 已经在 PROJECT_ROOT 下），兜底打印
                    image_path_str = abs_img_path.as_posix()
                    print(f"[extract_figures] WARNING: image not under PROJECT_ROOT, stored abs path: {image_path_str}")

                print(f"[extract_figures] page={page_index + 1}, img_abs={abs_img_path}, stored_rel={image_path_str}")

                # 阶段 B&C：调用视觉模型做物理语义对标（带强兜底）
                caption, linked_params = self._annotate_figure_with_vlm(
                    abs_img_path, page_index + 1, param_summary_text
                )

                figures.append({
                    "id": f"page-{page_index + 1}",
                    "caption": caption,
                    "page": page_index + 1,
                    "linked_parameters": linked_params,
                    "image_path": image_path_str
                })

            doc.close()
        except Exception as e:
            print(f"⚠️ extract_figures 发生异常: {e}")

        return figures

    def _annotate_figure_with_vlm(self, abs_img_path: Path, page_index: int, param_summary_text: str):
        """
        使用视觉大模型（Qwen-VL 等）为单页图像生成物理语义说明和参数关联。

        为了兼容当前环境：
        - 如果调用失败或超时，会回退到简单的占位式 caption，linked_parameters 为空列表。
        - 具体的 VLM 调用细节可能需根据实际 DashScope / OpenAI 接口稍作调整。
        """
        default_caption = f"自动导出的第 {page_index} 页整体快照（待视觉模型细化为关键图表）"
        fallback = (default_caption, [])

        try:
            prompt = f"""
你是一名复杂等离子体物理学家和图像理解助手。现在给你一张论文中的图像（来自第 {page_index} 页）以及这篇论文中已经提取的物理参数列表。

【物理参数列表】（每行为一个参数）：
{param_summary_text}

【任务】请完成以下两点，并严格按照 JSON 格式回答：
1. 用 1 句话（不超过 40 个中文字符）说明这张图主要展示了什么物理现象或参数关系，特别是与哪些参数有关（例如：展示了随 M_T^2 增大链状结构形成的演化趋势）。
2. 从参数列表中挑出与该图最相关的 1–3 个参数，返回它们的“符号 symbol”或“名称 name”（原样返回即可）。

【输出格式】（严格 JSON，不要包含任何多余说明）：
{{
  "caption": "一句话物理说明",
  "linked_parameters": ["符号或名称1", "符号或名称2"]
}}
"""
            # 使用 base64 data URI 方式直接传输图像，避免远程无法访问本地路径的问题
            try:
                with open(abs_img_path, "rb") as f:
                    b64_bytes = base64.b64encode(f.read())
                    b64_str = b64_bytes.decode("ascii")
                data_uri = f"data:image/png;base64,{b64_str}"
            except Exception as e:
                print(f"❌ [VLM] 本地图像读取/base64 编码失败(page={page_index}): {repr(e)}")
                return fallback

            # 按 DashScope OpenAI 兼容多模态规范：content 为若干个 {type: \"text\"|\"image_url\", ...}
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": data_uri,
                            },
                        },
                    ],
                }
            ]

            print(f"[VLM] annotating figure page={page_index}, path={abs_img_path}, via=base64")
            try:
                vlm_response = client.chat.completions.create(
                    model="qwen-vl-max",  # 或其他可用视觉模型，如 qwen-vl-plus
                    messages=messages,
                    temperature=0.1,
                )
            except Exception as e:
                print(f"❌ [VLM] qwen-vl 调用失败(page={page_index}): {repr(e)}")
                return fallback

            raw_content = vlm_response.choices[0].message.content
            if isinstance(raw_content, list):
                # 兼容部分 SDK 会返回 content 为多段的情况
                text_parts = [c.get("text", "") for c in raw_content if isinstance(c, dict)]
                raw_text = "\n".join(text_parts).strip()
            else:
                raw_text = str(raw_content).strip()

            # 剥离可能的 ```json 包裹
            if "```json" in raw_text:
                raw_text = raw_text.split("```json", 1)[1].split("```", 1)[0].strip()
            elif "```" in raw_text:
                raw_text = raw_text.split("```", 1)[1].split("```", 1)[0].strip()

            try:
                parsed = json.loads(raw_text)
            except Exception as e:
                print(f"⚠️ [VLM] 解析 JSON 失败(page={page_index}): {repr(e)}; raw_text={raw_text}")
                return fallback

            caption = parsed.get("caption", "").strip() or default_caption
            linked = parsed.get("linked_parameters", []) or []

            # 规范化 linked_parameters 内容为字符串列表
            norm_linked = []
            for item in linked:
                if isinstance(item, str):
                    norm_linked.append(item.strip())
                elif isinstance(item, dict):
                    # 如果模型返回了带字段的对象，尝试取 name 或 symbol
                    val = item.get("symbol") or item.get("name")
                    if val:
                        norm_linked.append(str(val).strip())

            print(f"[VLM] page={page_index}, caption={caption}, linked={norm_linked}")
            return caption, norm_linked

        except Exception as e:
            print(f"⚠️ [VLM] annotate 发生异常(page={page_index}): {repr(e)}")
            return fallback

    def extract_paper_structure(self, file_path):
        """
        第一步：双模型流水线提取
        Stage 1: qwen-long 负责深度物理理解 (文本输出)
        Stage 2: qwen-turbo 负责严格 JSON 格式化
        """
        print(f"🚀 [阶段 1] 正在调用 qwen-long 进行深度物理提取: {file_path}")
        try:
            file_object = client.files.create(file=Path(file_path), purpose="file-extract")
        except Exception as e:
            print(f"❌ [阶段 1] file-extract 失败: {repr(e)}")
            raise

        # --- Stage 1: qwen-long 提示词 (侧重物理与理解) ---
        extraction_prompt = """
        你是一个物理学专家。请阅读论文，提取核心信息并按以下【标签格式】输出。
        注意：不要输出JSON，不要输出Markdown，直接输出标签和内容。
        每一个参量和力场请按列表形式列出。

        输出规范：
        [metadata.title]: 标题
        [metadata.journal]: 期刊
        [metadata.year]: 年份
        [metadata.innovation]: 创新点
        [physics_context.environment]: 实验环境
        [physics_context.detailed_background]: 背景描述
        [observed_phenomena]: 观察到的物理现象
        [simulation_results_description]: 模拟结果描述
        [keywords]: 关键词1, 关键词2
        [experiment_setup]: 实验装置描述

        [parameter]: 
        名称: | 符号: | 数值: | 单位: | 含义: | 富化物理意义: | 来源: (原文/推断)

        [force_field]:
        名称: | 公式: | 物理本质: | 模拟计算建议(含单位):

        [interparticle_interaction]: 
        注意：此处只提取描述【微粒与微粒之间】相互作用的对势（Pair Potential）或力场,同时此力场将被用于模拟,所以只能有一个并且最为符合原有物理背景。
        - 严禁包含：外加电场（External AC/DC Field）、重力、磁场、整体限制力等背景参数。
        格式：名称: | 公式: | 物理本质: | 模拟计算建议(含单位):
        """

        # 第一步调用
        try:
            extraction_response = client.chat.completions.create(
                model="qwen-long",
                messages=[
                    {'role': 'system', 'content': f'fileid://{file_object.id}'},
                    {'role': 'system', 'content': extraction_prompt},
                    {'role': 'user', 'content': '请按格式提取论文内容。'}
                ],
                temperature=0.1,
            )
        except Exception as e:
            print(f"❌ [阶段 1] qwen-long 调用失败: {repr(e)}")
            # 向上抛出，让前端看到具体错误信息（app3 会展示在“解析失败: ...”里）
            raise

        try:
            extracted_text = extraction_response.choices[0].message.content.strip()
        except Exception as e:
            print(f"❌ [阶段 1] 解析 qwen-long 返回内容失败: {repr(e)}; raw_response={extraction_response}")
            raise

        print("✅ [阶段 1] 提取完成，进入阶段 2 格式化...")

        # --- Stage 2: qwen-turbo 提示词 (侧重格式与校验) ---
        formatting_prompt = """
        你是一个严格的JSON转换助手。
        任务：将用户提供的【标签化物理数据】转换为严格的JSON格式。

        要求：
        1. 严格遵守以下JSON结构。
        2. 确保单位(unit)和数值(value)分离。
        3. 禁止输出任何解释文字，只输出JSON主体。
        4. 禁止 trailing commas。
        5. 物理公式使用 Latex 语法。
           - 对于 force_fields[i].formula 字段：只填写【单个 LaTeX 公式本体】，不要在字符串中再包裹 $ 或 $$，也不要加入额外说明文字。
           - 示例：正确: "W(r,\\theta) = \\frac{Q^2}{r} e^{-r/\\lambda}"；错误: "$ W(r,\\theta) = ... $" 或 "W(...) = ..., 其中 Q 表示电荷"。
        6. 针对 force_fields 字段：只保留【微粒间相互作用势（Pair Potentials）】。如果输入中包含“外加电场”或“背景场”，请将其物理参数归类到 parameters 中，严禁放入 force_fields。

        目标结构（请严格遵守字段名和层级；除 formula 字段外，其他包含数学符号的地方可以使用内嵌 $...$ 以便前端渲染）：
        {
            "metadata": {"title": "", "journal": "", "year": "", "innovation": ""},
            "physics_context": {"environment": "", "detailed_background": ""},
            "observed_phenomena": "",
            "simulation_results_description": "",
            "keywords": [],
            "parameters": [
                {
                    "name": "", "symbol": "", "value": "", "unit": "",
                    "meaning": "", "enriched_physics": "", "source": ""
                }
            ],
            "force_fields": [
                {
                    "name": "", "formula": "",
                    "physical_significance": "", "computational_hint": ""
                }
            ],
            "experiment_setup": "",
            "figures": [
                {
                    "id": "",                 
                    "caption": "",            
                    "page": 0,                
                    "linked_parameters": [],  
                    "image_path": ""         
                }
            ]
        }
        """

        # 第二步调用：使用更擅长格式化的 qwen-turbo (或 qwen-plus)
        try:
            format_response = client.chat.completions.create(
                model="qwen-plus",
                messages=[
                    {'role': 'system', 'content': formatting_prompt},
                    {'role': 'user', 'content': f"请将以下内容转换为JSON：\n\n{extracted_text}"}
                ],
                temperature=0,  # 极低温度确保稳定性
            )
        except Exception as e:
            print(f"❌ [阶段 2] qwen-plus 调用失败: {repr(e)}")
            raise

        try:
            raw_json = format_response.choices[0].message.content.strip()
        except Exception as e:
            print(f"❌ [阶段 2] 解析 qwen-plus 返回内容失败: {repr(e)}; raw_response={format_response}")
            raise

        # --- 安全解析与兜底逻辑 ---
        # 模板定义（新增 figures 字段，保证下游结构一致）
        default_structure = {
            "metadata": {
                "title": f"解析失败_{Path(file_path).name}",
                "journal": "Unknown",
                "year": "Unknown",
                "innovation": "None"
            },
            "physics_context": {"environment": "Unknown", "detailed_background": "None"},
            "observed_phenomena": "None",
            "simulation_results_description": "None",
            "keywords": [],
            "parameters": [],
            "force_fields": [],
            "experiment_setup": "None",
            "figures": []
        }

        # 剥离代码块
        if "```json" in raw_json:
            raw_json = raw_json.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_json:
            raw_json = raw_json.split("```")[1].split("```")[0].strip()

        try:
            structured_data = json.loads(raw_json)
            # 深度合并确保不丢失Key
            import copy
            final_data = copy.deepcopy(default_structure)
            # 简单的逻辑合并
            for k, v in structured_data.items():
                if isinstance(v, dict) and k in final_data and isinstance(final_data[k], dict):
                    final_data[k].update(v)
                else:
                    final_data[k] = v

            # 调用图像提取钩子，补充 figures 字段
            try:
                figures = self.extract_figures(file_path, final_data)
                if figures:
                    final_data["figures"] = figures
            except Exception as e:
                print(f"⚠️ 提取图像失败: {e}")

            print(final_data)
            return final_data

        except json.JSONDecodeError as e:
            print(f"❌ JSON 转换阶段失败: {e}")
            # 保底方案：从 extracted_text 中正则抢救一个标题
            import re
            title_match = re.search(r'\[metadata\.title\]:\s*(.*)', extracted_text)
            if title_match:
                default_structure["metadata"]["title"] = title_match.group(1).strip()
            return default_structure

    def search_knowledge(self, query_text, top_k=2):
        """基于向量检索结果，从 SQLite 硬盘数据库回捞详尽元数据"""
        query_vector = self.get_embedding(query_text)

        D1, I1 = self.paper_index.search(query_vector, top_k)
        D2, I2 = self.force_index.search(query_vector, top_k)

        relevant_papers = []
        relevant_forces = []

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # 从 I1 (论文向量 ID 列表) 回捞
            for v_id in I1[0]:
                if v_id == -1: continue
                cursor.execute("SELECT metadata_json FROM papers WHERE vector_id = ?", (int(v_id),))
                res = cursor.fetchone()
                if res: relevant_papers.append(json.loads(res[0]))

            # 从 I2 (力场向量 ID 列表) 回捞
            for v_id in I2[0]:
                if v_id == -1: continue
                cursor.execute("SELECT force_json, source_paper FROM force_fields WHERE vector_id = ?", (int(v_id),))
                res = cursor.fetchone()
                if res:
                    f_data = json.loads(res[0])
                    f_data['source_from'] = res[1]  # 附带来源信息
                    relevant_forces.append(f_data)

        return relevant_papers, relevant_forces

    def get_simulation_recommendation(self, structured_paper, user_params):
        """
        第二步：动态针对用户提供的每个参数（及其物理含义）给出推荐
        """
        # RAG 检索逻辑
        # 增加安全获取逻辑，防止 keywords 中混入 int / float 导致 join 报错
        title = structured_paper.get('metadata', {}).get('title', 'Unknown')
        raw_keywords = structured_paper.get('keywords', [])
        if isinstance(raw_keywords, (list, tuple, set)):
            keywords = [str(k) for k in raw_keywords]
        elif raw_keywords:
            keywords = [str(raw_keywords)]
        else:
            keywords = []
        search_query = f"{title} " + " ".join(keywords)
        relevant_papers, relevant_forces = self.search_knowledge(search_query)

        # 修改点：提取参数名列表（此时 user_params 是嵌套字典）
        # 获取用户期望现象
        expected_phenomena = user_params.get("expected_phenomena", "无")
        param_list_str = ", ".join([k for k in user_params.keys() if k != "expected_phenomena"])

        # 修改点：Prompt 强化了对物理含义（description）的关注
        prompt = f"""
        你现在是一名复杂等离子体物理模拟专家。

        【物理参考上下文】：
        - 参考文献核心结构：{json.dumps(structured_paper, indent=2)}
        - 关联力场库知识：{json.dumps(relevant_forces, indent=2)}

        【参考论文物理现象】：
        - 实验观察：{structured_paper.get('observed_phenomena')}
        - 模拟表现：{structured_paper.get('simulation_results_description')}
        - 参考力场：{json.dumps(relevant_forces, indent=2)}

        【用户模拟需求】：
        - 待模拟参数：{json.dumps({k: v for k, v in user_params.items() if k != 'expected_phenomena'}, indent=2)}
        - 期望观察到的现象：{expected_phenomena}

        【任务指令】：
        请针对上述每个参数，结合其【物理含义说明】和参考文献中的实验/理论背景，进行精确的区间推荐：
        现象匹配：根据用户期望观察到的【{expected_phenomena}】，结合参考论文中的现象描述，调整推荐的参数区间。
        1. 推荐区间 [Min, Max]：必须符合物理描述中的时空尺度要求。
        2. 建议步长 (Step Size)：必须足以捕捉物理描述中提到的关键特征（如波形解析、碰撞频率等）。
        3. 深度理由：请引用参考文献中的公式或物理常数（如等离子体频率 ωpd）来支撑你的区间选择。
        4. 单位核心原则：所有推荐的区间 [Min, Max] 和步长 val 必须严格匹配物理单位。如果用户输入的是 'ms'，推荐也必须以 'ms' 为基准，严禁单位混乱。  

        此外，请根据物理含义推荐一个最匹配的【模拟力场模型】。

        【输出格式要求】（严格 JSON）：
        {{
          "parameter_recommendations": {{
             "参数名": {{
                 "range": [min, max], 
                 "step": val, 
                 "unit": "物理单位", 
                 "reason": "结合现象匹配和单位约束的理由"
             }}
          }},
          "force_field_recommendation": {{
             "name": "力场名称",
             "reason": "为什么该力场能复现期望现象的物理依据"
          }}
        }}
        """

        completion = client.chat.completions.create(
            model="qwen-long",
            messages=[
                {'role': 'system', 'content': '你是一个精通复杂等离子体物理和数值算法的资深科学家。请直接输出JSON。'},
                {'role': 'user', 'content': prompt}
            ]
        )

        return completion.choices[0].message.content

    def get_embedding(self, text):
        """调用阿里云配套 Embedding 模型"""
        # 注意：这里需要对输入文本进行简单处理，确保它是字符串且不为空
        if not text or not text.strip():
            text = "empty_input_placeholder"  # 防止空字符串导致API报错
        text = text.replace("\n", " ")

        response = client.embeddings.create(
            model="text-embedding-v2",  # 配套的向量模型
            input=text
        )
        # 将结果转为 float32 的 numpy 数组，并增加一个维度 (1, 1536) 以适配 FAISS
        return np.array(response.data[0].embedding).astype('float32').reshape(1, -1)

    def _is_valid_physics_data(self, data):
        """
        质量校验逻辑：判断提取的数据是否具备物理研究价值
        """
        # 1. 标题校验
        title = data.get('metadata', {}).get('title', "")
        if not title or "解析失败" in title or title == "Unknown":
            return False

        # 2. 核心物理内容校验
        # 如果 parameters 和 force_fields 同时为空，说明没提取到任何关键物理建模信息
        if not data: return False
        params = data.get('parameters', [])
        forces = data.get('force_fields', [])
        if not params and not forces:
            print("⚠️ 质量校验失败：未提取到任何物理参数或力场信息。")
            return False

        # 3. 文本富化程度校验
        # 如果背景描述和创新点都是默认的 "None" 或 "Unknown"，说明理解失败
        innovation = data.get('metadata', {}).get('innovation', "None")
        background = data.get('physics_context', {}).get('detailed_background', "None")
        if innovation in ["None", "Unknown"] and background in ["None", "Unknown"]:
            print("⚠️ 质量校验失败：物理背景理解为空。")
            return False

        return True

    def _safe_save_index(self, index, path):
        """原子化保存 FAISS 索引"""
        tmp_path = path + ".tmp"
        try:
            faiss.write_index(index, tmp_path)
            # os.replace 是原子的，确保文件完整性
            os.replace(tmp_path, path)
        except Exception as e:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise IOError(f"保存索引失败: {e}")

    def update_vector_db(self, structured_data):
        """
        持久化更新：查重 -> 写入SQLite -> 写入FAISS -> 同步到磁盘文件
        """
        title = structured_data['metadata']['title']

        # 1. 深度质量校验
        if not self._is_valid_physics_data(structured_data):
            print(f"❌ 数据质量未达标，不记录到数据库。请检查论文格式或 API 配额。")
            return

        # 使用 portalocker 给索引文件加锁，防止多进程同时写入导致文件损坏
        # 我们创建一个 .lock 文件作为信号灯
        lock_path = self.db_path + ".lock"

        try:
            with portalocker.Lock(lock_path, timeout=10):
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    # 开启显式事务：确保数据库和索引文件要么都成功，要么都失败
                    conn.execute("BEGIN")

                    # --- 1. 论文查重与写入 ---
                    cursor.execute("SELECT id FROM papers WHERE title = ?", (title,))
                    if cursor.fetchone():
                        print(f"跳过已存在论文: {title}")
                        return
                    else:
                        # paper_text = f"Title: {title}. Context: {structured_data['physics_context']['detailed_background']}"
                        background = structured_data.get('physics_context', {}).get('detailed_background',
                                                                                    'No background available')
                        paper_text = f"Title: {title}. Context: {background}"
                        paper_vec = self.get_embedding(paper_text)

                        # 写入 FAISS 内存并获取当前的索引位置
                        # vector_id = self.paper_index.ntotal
                        # self.paper_index.add(paper_vec)

                        # 写入 SQLite 和磁盘文件
                        # cursor.execute("INSERT INTO papers (title, metadata_json, vector_id) VALUES (?, ?, ?)",
                        #                (title, json.dumps(structured_data), vector_id))

                        # 1. 先存数据库，拿到自增 ID
                        # 使用自增的显式 id（如从 SQLite 获得的 paper row id，或用 time-based int id）
                        cursor.execute("INSERT INTO papers (title, metadata_json, vector_id) VALUES (?, ?, ?)",
                                       (title, json.dumps(structured_data), -1))
                        paper_row_id = cursor.lastrowid

                        # 2. 把这个 ID 同步给 FAISS
                        # 用 paper_row_id 作为外部向量 id
                        self.paper_index.add_with_ids(paper_vec, np.array([paper_row_id], dtype='int64'))

                        # 3. 更新数据库中的 vector_id 字段
                        # 然后更新 DB 的 vector_id 字段为 paper_row_id
                        cursor.execute("UPDATE papers SET vector_id = ? WHERE id = ?",
                                       (int(paper_row_id), paper_row_id))

                        # 4. 立即写回磁盘
                        # faiss.write_index(self.paper_index, self.paper_idx_path)
                        self._safe_save_index(self.paper_index, self.paper_idx_path)
                        print(f"✅ 论文已存入: {title} (ID: {paper_row_id})")

                    # --- 2. 新增：图片信息入库 ---
                    if "figures" in structured_data:
                        for fig in structured_data["figures"]:
                            print(
                                f"[update_vector_db] storing figure for paper={title}, image_path={fig.get('image_path')}")
                            cursor.execute('''
                                INSERT INTO figures (paper_id, image_path, caption, page_num)
                                VALUES (?, ?, ?, ?)
                            ''', (
                                paper_row_id,
                                fig.get("image_path"),
                                fig.get("caption"),
                                fig.get("page")
                            ))
                        print(f"✅ 图片已存入: {title} (ID: {paper_row_id})")

                    # --- 2. 力场查重与写入 ---
                    if "force_fields" in structured_data:
                        for ff in structured_data["force_fields"]:
                            # 生成唯一特征哈希：公式 + 物理环境
                            # formula_str = ff['formula'] + structured_data['physics_context']['environment']
                            # 修改后（更安全）
                            formula_val = ff.get('formula', 'unknown_formula')
                            env_val = structured_data.get('physics_context', {}).get('environment', 'unknown_env')
                            formula_str = formula_val + env_val

                            f_hash = hashlib.md5(formula_str.encode()).hexdigest()

                            cursor.execute("SELECT id FROM force_fields WHERE formula_hash = ?", (f_hash,))
                            if cursor.fetchone():
                                continue  # 相似背景下的相同公式，跳过

                            force_feature = f"Interparticle Interaction: {ff['name']}. Significance: {ff['physical_significance']}"
                            force_vec = self.get_embedding(force_feature)

                            # f_vector_id = self.force_index.ntotal
                            # self.force_index.add(force_vec)
                            #
                            # cursor.execute(
                            #     "INSERT INTO force_fields (formula_hash, force_json, source_paper, vector_id) VALUES (?, ?, ?, ?)",
                            #     (f_hash, json.dumps(ff), title, f_vector_id))

                            # 1. 先存数据库拿到 ID
                            cursor.execute(
                                "INSERT INTO force_fields (formula_hash, force_json, source_paper, vector_id) VALUES (?, ?, ?, ?)",
                                (f_hash, json.dumps(ff), title, -1))
                            db_force_id = cursor.lastrowid

                            # 2. 同步 ID 到力场索引
                            self.force_index.add_with_ids(force_vec, np.array([db_force_id], dtype='int64'))

                            # 3. 回填 vector_id
                            cursor.execute("UPDATE force_fields SET vector_id = ? WHERE id = ?",
                                           (db_force_id, db_force_id))

                        # faiss.write_index(self.force_index, self.force_idx_path)
                        self._safe_save_index(self.force_index, self.force_idx_path)

                    conn.commit()

        except portalocker.exceptions.LockException:
            print("❌ 无法获取文件锁，可能有其他进程正在写入。")
        except Exception as e:
            print(f"❌ 数据库更新发生错误: {e}")
            # 这里自动触发 rollback (因为在 with sqlite3.connect 块内)


if __name__ == "__main__":
    rag_system = ComplexPlasmaRAG()

    # 用户输入：增加期望现象，并强化单位意识
    user_input_params = {
        "target_particle_charge": {
            "value": "1.2 * 10^4",
            "unit": "e",
            "description": "单个尘埃微粒携带的电荷量。"
        },
        "time_scale": {
            "value": "200.0",
            "unit": "ms",
            "description": "模拟演化的总时长。"
        },
        "debye_length_target": {
            "value": "0.6",
            "unit": "mm",
            "description": "系统预期的德拜屏蔽长度。"
        },
        "expected_phenomena": "观察到微粒在微重力流场中由于非对称相互作用形成的链状结构(string formation)。"
    }

    # 执行流程
    pdf_path = r"/getdata/output/Complex_Plasma_Simulation/pdfs/MD—Ivlev_PRL_2008.pdf"
    structured_json = rag_system.extract_paper_structure(pdf_path)

    # 持久化更新（此时内部已包含现象描述和推断标注）
    rag_system.update_vector_db(structured_json)

    # 获取包含现象匹配建议的推荐
    recommendation = rag_system.get_simulation_recommendation(structured_json, user_input_params)

    print("\n--- 模拟参数推荐结果 (现象对标版) ---")
    print(recommendation)