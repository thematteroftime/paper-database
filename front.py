# frontend_refined.py
"""
PlasmaRAG Frontend Application
Version: 1.0.1
Compatible with Gradio 6.5.1
"""
import os
import json
import sqlite3
import base64
import tempfile
import pathlib
import html as html_escape
import re
import dashscope
import gradio as gr
import pandas as pd
from backend import ComplexPlasmaRAG

# Version information
__version__ = "1.0.1"

# Base directory of this app (used to resolve demo asset paths like images/)
BASE_DIR = pathlib.Path(__file__).resolve().parent


# ---- 路径相关小工具 ----
def normalize_figure_path(path: str) -> str:
    """
    将任意形式的路径统一规范为【绝对路径 + 正斜杠】，便于 Gradio 的 file= 协议访问。

    核心思路：
    - 统一转为以 BASE_DIR 为基准的绝对路径（即当前前端脚本所在目录）；
    - 使用 Path.resolve() 明确真实位置，避免工作目录变化带来的相对路径偏移；
    - 将 Windows 的 "\\" 变成 "/"，与浏览器 / Gradio 的 /file= 协议兼容。
    """
    if not path:
        return ""
    path_str = str(path)
    if path_str.startswith(("http://", "https://", "data:")):
        return path_str
    try:
        p = pathlib.Path(path_str)
        # 如果是绝对路径且在项目目录下，转换为相对 BASE_DIR 的路径
        if p.is_absolute():
            try:
                p = p.resolve().relative_to(BASE_DIR)
            except ValueError:
                # 不在项目目录内时，退化为仅使用文件名，避免跨盘符问题
                p = pathlib.Path(p.name)
        else:
            # 相对路径：以 BASE_DIR 为基准解析，再转回相对路径，保证规范化
            p = (BASE_DIR / p).resolve().relative_to(BASE_DIR)
        normalized = p.as_posix()
        print(f"[normalize_figure_path] raw={path_str}, normalized_rel={normalized}, exists={(BASE_DIR / p).exists()}")
        return normalized
    except Exception as e:
        print(f"[normalize_figure_path] 路径转换失败: {e}, raw={path_str}")
        return path_str


def extract_figure_paths(structured_data):
    """从结构化数据中提取用于 Gallery 展示的图片及文字说明列表。

    返回格式为 [[相对路径, 说明文本], ...]，以便在 Gallery 中像论文图注一样显示。
    """
    if not structured_data:
        return []
    figs = structured_data.get("figures", []) or []
    paths = []
    for f in figs:
        raw = f.get("image_path", "")
        if not raw:
            continue
        norm = normalize_figure_path(raw)
        if not norm:
            continue
        full_path = (BASE_DIR / norm).resolve()
        if full_path.is_file():
            # 组合图注：优先使用 caption，其次可附带页码信息
            caption = f.get("caption", "") or ""
            page = f.get("page", None)
            if page is not None:
                if caption:
                    caption_text = f"Page {page} · {caption}"
                else:
                    caption_text = f"Page {page}"
            else:
                caption_text = caption
            paths.append([norm, caption_text])
        else:
            print(f"[extract_figure_paths] 跳过非文件路径: raw={raw}, norm={norm}, full={full_path}")
    return paths


# ---- 调试用内置示例数据（便于前端渲染测试，无需调用后端） ----
DEMO_STRUCTURED_DATA = {
    'metadata': {
        'title': 'First Observation of Electrorheological Plasmas',
        'journal': 'Physical Review Letters',
        'year': '2008',
        'innovation': '首次发现“电变流变复杂等离子体”（ER complex plasmas），揭示了通过外加交流电场调控尘埃粒子间相互作用的新机制，并观察到从各向同性流体到链状（string）结构的可逆相变；该系统在微重力条件下实现，且粒子动力学可单粒子分辨，为研究ER流体的基本动力学过程提供了全新平台。'
    },
    'physics_context': {
        'environment': '微重力环境（国际空间站内），低气压氩气放电等离子体，含有带负电的微米级尘埃颗粒',
        'detailed_background': '复杂等离子体中的尘埃颗粒周围存在由补偿离子构成的“德拜球”（Debye sphere）。在外加交流电场作用下，离子漂移导致德拜球变形，形成非对称的“离子尾”（ion wake），从而诱导出偶极型相互作用。当电场频率远高于尘埃响应频率但低于离子响应频率时，时间平均后的有效相互作用是可逆的（Hamiltonian），适用于统计物理方法分析。'
    },
    'observed_phenomena': '随着外加交流电场强度增加，尘埃系统发生从各向同性的流体态到沿电场方向排列的一维链状结构（string phase）的相变；该相变是可逆的，降低电场后系统恢复初始状态；链状结构的形成趋势随粒子尺寸增大而增强。',
    'simulation_results_description': '分子动力学模拟显示，在弱耦合条件下，随着热马赫数 $ M_T $ 增大，纵向标度指数分布向更小值偏移，表明出现一维有序结构；横向与纵向标度指数差 $ \\Delta\\alpha $ 作为序参量，随 $ M_T $ 呈幂律增长，支持二级或弱一级相变；模拟结果与实验观测高度一致。',
    'keywords': [
        '电变流变等离子体', '尘埃等离子体', '相变', '链状结构', '德拜屏蔽', '离子尾', '分子动力学模拟'
    ],
    # Demo figures: use three local images in the images/ folder
    'figures': [
        {
            'id': 'demo_fig_1',
            'caption': '示例图 1：链状结构形成过程的示意，可用于测试“关键图表”排版效果。',
            'page': 1,
            'linked_parameters': ['d', 'M_T^2'],
            # 使用 raw string，避免反斜杠转义问题；后续会通过 normalize_figure_path 统一为绝对路径
            'image_path': r'images/image1.png',
        },
        {
            'id': 'demo_fig_2',
            'caption': '示例图 2：不同粒径下的相图或力场分布，用于检查多图横向滚动体验。',
            'page': 2,
            'linked_parameters': ['p', 'U_{pp}'],
            'image_path': r'images/image2.png',
        },
        {
            'id': 'demo_fig_3',
            'caption': '示例图 3：实验装置或时序示意，用于测试长 caption 的展示效果。',
            'page': 3,
            'linked_parameters': ['\\kappa', '\\Gamma'],
            'image_path': r'images/image3.png',
        },
    ],
    'parameters': [
        {
            'name': '粒子直径',
            'symbol': '$ d $',
            'value': '1.55, 4.9, 6.8',
            'unit': 'μm',
            'meaning': '实验所用微粒的几何直径',
            'enriched_physics': '决定粒子表面积和电荷收集能力，影响感应偶极矩大小',
            'source': '原文'
        },
        {
            'name': '气体压力',
            'symbol': '$ p $',
            'value': '8–15',
            'unit': 'Pa',
            'meaning': '氩气工作气压',
            'enriched_physics': '控制中性气体密度，进而影响离子迁移率和碰撞频率',
            'source': '原文'
        },
        {
            'name': '交流电频率',
            'symbol': '$ f $',
            'value': '100',
            'unit': 'Hz',
            'meaning': '施加于电极的AC信号频率',
            'enriched_physics': '满足 $ \\omega_{\\text{dust}} \\ll \\omega \\ll \\omega_{\\text{ion}} $ 条件，确保离子瞬时响应而尘埃不响应',
            'source': '原文'
        },
        {
            'name': '峰峰值电压',
            'symbol': '$ U_{pp} $',
            'value': '26.6–65.6',
            'unit': 'V',
            'meaning': '电极间施加的AC电压幅值',
            'enriched_physics': '决定电场强度，控制离子振荡速度和wake变形程度',
            'source': '原文'
        },
        {
            'name': '屏蔽长度',
            'symbol': '$ \\lambda $',
            'value': '~0.05',
            'unit': 'mm',
            'meaning': '等离子体对电场的屏蔽特征长度',
            'enriched_physics': '决定了粒子间相互作用的作用范围',
            'source': '推断（文中估计）'
        },
        {
            'name': '粒子电荷',
            'symbol': '$ Q $',
            'value': '~$-10^4$',
            'unit': '$ e $',
            'meaning': '尘埃颗粒携带的电子电荷数量',
            'enriched_physics': '主导德拜-休克尔排斥力的强度',
            'source': '原文（实验估算）'
        },
        {
            'name': '数密度',
            'symbol': '$ n $',
            'value': '~$ 3 \\times 10^4 $',
            'unit': 'cm⁻³',
            'meaning': '单位体积内的尘埃粒子数目',
            'enriched_physics': '影响平均间距和耦合强度',
            'source': '原文（实验估算）'
        },
        {
            'name': '热马赫数平方',
            'symbol': '$ M_T^2 $',
            'value': '0.22–1.45',
            'unit': '无量纲',
            'meaning': '离子振荡速度相对于热速度的比值平方',
            'enriched_physics': '核心控制参数，决定wake变形程度和偶极相互作用强度',
            'source': '原文（通过模拟反推）'
        },
        {
            'name': '屏蔽参数',
            'symbol': '$ \\kappa $',
            'value': '~7.7',
            'unit': '无量纲',
            'meaning': '$ \\kappa = \\Delta / \\lambda $，其中 $ \\Delta = n^{-1/3} $ 为平均间距',
            'enriched_physics': '表征系统的屏蔽强弱，用于相图绘制',
            'source': '原文（模拟设定）'
        },
        {
            'name': '耦合参数',
            'symbol': '$ \\Gamma $',
            'value': '530 或 133',
            'unit': '无量纲',
            'meaning': '$ \\Gamma = Q^2 / (\\lambda T) $，表示静电能与热动能之比',
            'enriched_physics': '判断系统是否处于强耦合或弱耦合状态',
            'source': '原文（模拟输入）'
        },
    ],
    'force_fields': [
        {
            'name': '时间平均后的有效对势',
            'formula': '$ W(r,\\theta) = \\frac{Q^2}{r} e^{-r/\\lambda} \\left[ 1 + 0.43 M_T^2 \\frac{\\lambda^2}{r^2} (3\\cos^2\\theta - 1) \\right] $',
            'physical_significance': '包含德拜-休克尔核心项与电场诱导的四极（等效偶极）修正项，源于离子尾的时间平均效应',
            'computational_hint': '可作为静态有效势用于分子动力学模拟；距离 $ r $ 和 $ \\lambda $ 单位统一为 mm 或 μm；角度 $ \\theta $ 为相对电场方向夹角；$ M_T $ 为无量纲控制参数'
        }
    ],
    'experiment_setup': '实验在国际空间站上的“PK-3 Plus”实验室进行，使用两块平行水平射频电极施加100 Hz正弦异相信号（峰峰值电压26.6–65.6 V）；氩气压力8–15 Pa；注入不同尺寸（1.55 μm, 4.9 μm, 6.8 μm）的微粒以调节数密度；利用薄激光片照明并记录粒子三维位置演化'
}

# ---- Simulation Setup 视图推荐卡片的调试用示例 JSON ----
DEMO_RECOMMENDATION_JSON = {
    "parameter_recommendations": {
        "target_particle_charge": {
            "range": [10000.0, 15000.0],
            "step": 500.0,
            "unit": "e",
            "reason": "参考文献中粒子电荷量约为 ~−10⁴ e（见 parameters[4]），且链状结构形成趋势随 |Q| 增大而增强（observed_phenomena）；结合库仑耦合参数 Γ ∝ Q²/λ，为确保强耦合（Γ > 100，对应液体/有序态），同时避免数值发散（Q过大会导致短程斥力爆炸），推荐区间覆盖实验典型值并略作上扩；单位严格匹配输入 'e'。"
        },
        "time_scale": {
            "range": [150.0, 250.0],
            "step": 10.0,
            "unit": "ms",
            "reason": "微重力下尘埃等离子体动力学时间尺度由离子响应主导：特征时间 τ_i ≈ 1/ω_pi，其中 ω_pi ≈ √(n_i e²/(ε₀ m_i))；取 Ar 气 p = 10 Pa → n_i ≈ 2.5×10²⁰ m⁻³ → ω_pi ≈ 1.2×10⁶ rad/s → τ_i ≈ 0.8 μs；但尘埃运动受离子尾流调制，有效演化时间由马赫数 M_T 决定，M_T = v_d / c_s，c_s ≈ √(k_B T_e / m_i) ~ 1000 m/s，v_d ~ E/(ν_in) ~ U_pp/(p·d) ~ 10 cm/s ⇒ M_T ~ 10⁻⁴–10⁻¹；文献中相变在 M_T ∈ [0.22, 1.45] 显著发生（parameters[7]），对应动力学演化需覆盖 ≥100 倍粒子振荡周期（τ_d ≈ 2π√(m_d / (Q²/λ² ε₀)) ~ 1–10 ms）；故总模拟时长 200 ms 足以捕捉链形成与弛豫（见 figures[3][4] 中相变演化图），区间扩展 ±25% 保障统计收敛，步长 10 ms 可解析链生长动力学（如形核、合并事件）。单位严格为 'ms'。"
        },
        "debye_length_target": {
            "range": [0.4, 0.8],
            "step": 0.05,
            "unit": "mm",
            "reason": "文献中 λ ≈ 0.05 mm（parameters[5]），但该值对应低压（8–15 Pa）Ar 气及典型电子温度；用户指定 λ = 0.6 mm，比文献值大 12×，表明系统更稀薄或 T_e 更高；根据德拜长度定义 λ_D = √(ε₀ k_B T_e / (n_e e²))，增大 λ 需降低 n_e 或提高 T_e；为维持可观测的尾流各向异性（∝ M_T² λ²/r²），必须保证 κ = λ/Δ ≥ 5（parameters[8]，κ=7.7），即平均粒间距 Δ ≤ λ/5 = 0.12 mm；对应粒子数密度 n ≥ (1/Δ)³ ≈ 6×10⁵ cm⁻³ —— 此值高于文献 n≈3×10⁴ cm⁻³，但仍在 PK-3 Plus 可达范围（高功率放电可提升 n_e）；因此 λ=0.6 mm 是可行且有利于增强长程各向异性作用（W ∝ e⁻ʳ⁄λ），促进链稳定；区间 [0.4, 0.8] mm 覆盖弱至强屏效过渡，步长 0.05 mm 可分辨 λ 对序参量 Δα 的幂律依赖（simulation_results_description 中 Δα ∝ M_T^β）。单位严格为 'mm'。"
        }
    },
    "force_field_recommendation": {
        "name": "场致电变流体对势（Electrorheological Pair Potential）",
        "reason": "该力场显式包含各向异性项 −0.43 M_T² (3cos²θ−1)/(r/λ)，直接编码了外加交变电场下离子尾流诱导的偶极类相互作用（physical_significance），且其角依赖性（cos²θ）在 θ=0（沿电场方向）产生净吸引，驱动一维链状有序（observed_phenomena）；相比 '时间平均尾流势'，此势已是静态有效对势，可直接用于分子动力学模拟，无需实时求解等离子体响应；文献明确指出其适用于研究各向异性相变行为（computational_hint），且模拟结果与微重力实验高度吻合（simulation_results_description）；而 '时间平均尾流势' 仅描述单粒子势场，不满足粒子间相互作用建模需求。"
    }
}

# ---- 后端实例（不要在前端传 API key） ----
# ComplexPlasmaRAG 内部应读取环境变量 DASHSCOPE_API_KEY
MY_API_KEY = os.getenv("DASHSCOPE_API_KEY", "sk-fd7afdef962a46d39784e8b0b8133974")
rag_system = ComplexPlasmaRAG(api_key=MY_API_KEY)
dashscope.api_key = MY_API_KEY


# ---- 小工具 ----
def safe_json_load(s):
    try:
        return json.loads(s)
    except:
        return None


def render_progress_html(steps_done):
    # steps_done: list of bool states [upload, parsing, extraction, embedding, indexed]
    labels = ["Upload", "Parsing", "Physics Extraction", "Embedding", "Indexed"]
    html = '<div style="font-family:Inter, Arial, sans-serif;">'
    for i, lab in enumerate(labels):
        ok = steps_done[i]
        color = "#16a34a" if ok else "#9ca3af"
        sym = "✅" if ok else "○"
        html += f'<div style="margin:6px 0;"><span style="color:{color};font-weight:600;margin-right:8px">{sym}</span>{lab}</div>'
    html += "</div>"
    return html


def card_css():
    """全局样式：工作台布局 + 卡片 + 参数网格 + 力场卡片。

    仅包含 CSS，不包含脚本；数学公式渲染交给 Gradio 的 Markdown / KaTeX。
    """
    return """
    <style>
      :root {
        --paper-bg: #f9fafb;
        --card-bg: #ffffff;
        --card-border: #e5e7eb;
        --accent: #4f46e5;
        --accent-soft: rgba(79,70,229,0.06);
        --muted: #6b7280;
        --text-main: #111827;
      }
      .paper-workbench {
        background: var(--paper-bg);
        border-radius: 16px;
        padding: 18px 22px 22px 22px;
        box-shadow: 0 18px 45px rgba(15,23,42,0.06);
        display: flex;
        flex-direction: column;
        gap: 18px;
      }
      .paper-header {
        display: flex;
        flex-direction: column;
        gap: 4px;
      }
      .paper-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: var(--text-main);
      }
      .paper-meta-line {
        font-size: 0.88rem;
        color: var(--muted);
      }

      .pipeline {
        display: flex;
        flex-direction: column;
        gap: 6px;
        margin-top: 4px;
      }
      .pipeline-step {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.82rem;
        color: #4b5563;
      }
      .pipeline-badge {
        width: 16px;
        height: 16px;
        border-radius: 999px;
        border: 1.5px solid #22c55e;
        background: #22c55e;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 0.7rem;
      }

      .paper-main-grid {
        display: grid;
        grid-template-columns: minmax(0,2.1fr) minmax(0,2.4fr);
        gap: 16px;
        align-items: flex-start;
      }

      .paper-card {
        background: var(--card-bg);
        border-radius: 12px;
        border: 1px solid var(--card-border);
        padding: 14px 16px;
        margin-bottom: 6px;
      }
      .paper-card h3 {
        margin: 0 0 8px 0;
        font-size: 0.96rem;
        font-weight: 600;
        color: var(--text-main);
      }
      .paper-card p,
      .paper-card li {
        font-size: 0.85rem;
        color: var(--muted);
        line-height: 1.6;
      }

      .paper-card-physics {
        border-left: 3px solid var(--accent);
        padding-left: 13px;
      }

      .param-section-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 8px;
      }
      .param-section-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: var(--text-main);
      }
      .param-section-sub {
        font-size: 0.8rem;
        color: var(--muted);
      }

      .param-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
        gap: 10px;
      }
      .param-card {
        border-radius: 10px;
        border: 1px solid #e5e7eb;
        background: #f9fafb;
        padding: 9px 10px;
        display: flex;
        flex-direction: column;
        gap: 2px;
      }
      .param-symbol {
        font-size: 0.88rem;
        color: var(--muted);
        font-style: italic;
      }
      .param-value {
        font-size: 1.15rem;
        font-weight: 700;
        color: #111827;
      }
      .param-unit {
        font-size: 0.78rem;
        font-weight: 500;
        color: #6366f1;
      }
      .param-name {
        font-size: 0.82rem;
        font-weight: 500;
        color: #111827;
      }
      .param-meaning {
        font-size: 0.78rem;
        color: var(--muted);
      }

      .phenomena-card {
        margin-top: 6px;
        border-radius: 12px;
        padding: 12px 14px;
        background: linear-gradient(135deg,#eef2ff,#f9fafb);
        border: 1px solid #e0e7ff;
      }
      .phenomena-title {
        font-size: 0.94rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 4px;
      }
      .phenomena-body {
        font-size: 0.85rem;
        color: #334155;
        line-height: 1.7;
      }

      .force-section {
        margin-top: 10px;
        display: flex;
        flex-direction: column;
        gap: 10px;
      }
      .force-card {
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        background: #ffffff;
        padding: 10px 12px;
      }
      .force-name {
        font-size: 0.9rem;
        font-weight: 600;
        color: #111827;
        margin-bottom: 6px;
      }
      .force-formula {
        background: #f9fafb;
        border-radius: 8px;
        padding: 8px 10px;
        text-align: center;
        margin-bottom: 6px;
        font-size: 0.98rem;
      }
      .force-text {
        font-size: 0.8rem;
        color: #4b5563;
        line-height: 1.6;
      }

      /* Simulation Setup 参数表右上角加减号按钮 */
      .param-row-btn > button {
        min-width: 32px !important;
        max-width: 32px !important;
        height: 32px !important;
        padding: 0 !important;
        border-radius: 6px !important;
        font-size: 0.9rem !important;
      }

      /* Simulation Setup 推荐仪表盘卡片样式 */
      .recom-wrapper {
        display: flex;
        flex-direction: column;
        gap: 14px;
        margin-top: 8px;
      }
      .recom-card {
        background: #ffffff;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        border-left: 3px solid #4f46e5;
        padding: 12px 16px 12px 14px;
        display: flex;
        flex-direction: column;
        gap: 8px;
      }
      .recom-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 4px;
      }
      .recom-name {
        font-size: 0.95rem;
        font-weight: 600;
        color: #111827;
      }
      .unit-badge {
        font-size: 0.78rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 999px;
        background: #eef2ff;
        color: #4338ca;
      }
      .recom-values {
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
      }
      .value-slot {
        flex: 1 1 140px;
        min-width: 0;
      }
      .slot-label {
        font-size: 0.78rem;
        font-weight: 500;
        color: #6b7280;
        margin-bottom: 4px;
      }
      .slot-value {
        font-size: 0.9rem;
        font-weight: 600;
        color: #111827;
        word-break: break-all;
      }
      .range-display {
        font-size: 0.9rem;
        font-weight: 600;
        color: #1d4ed8;
        margin-bottom: 4px;
      }
      .range-track {
        position: relative;
        width: 100%;
        height: 4px;
        border-radius: 999px;
        background: #e5e7eb;
        overflow: hidden;
      }
      .range-fill {
        position: absolute;
        left: 8%;
        right: 8%;
        top: 0;
        bottom: 0;
        border-radius: 999px;
        background: linear-gradient(90deg,#4f46e5,#22c55e);
        opacity: 0.75;
      }
      .reason-box {
        margin-top: 6px;
        padding: 10px 12px;
        border-radius: 8px;
        background: #f0f7ff;
        border: 1px solid #bfdbfe;
        font-size: 0.85rem;
        color: #1e3a8a;
        line-height: 1.7;
        white-space: pre-wrap;
      }
      .recom-force-card {
        margin-top: 10px;
        padding: 14px 16px;
        border-radius: 12px;
        background: #eef2ff;
        border: 1px solid #c7d2fe;
        color: #111827;
      }
      .recom-force-title {
        font-size: 1.0rem;
        font-weight: 600;
        margin-bottom: 6px;
        color: #1e3a8a;
      }
      .recom-force-name {
        font-size: 0.95rem;
        font-weight: 600;
        color: #4338ca;
        margin-bottom: 6px;
      }
      .recom-force-body {
        font-size: 0.86rem;
        line-height: 1.7;
        color: #1f2937;
      }
      .recom-expert-json {
        margin-top: 16px;
        font-size: 0.8rem;
        background:#0f172a;
        color:#e5e7eb;
        padding:12px;
        border-radius:8px;
        overflow:auto;
        max-height:260px;
      }
    </style>
    """


# ---- 页面组件行为函数 ----

def process_pdf_step(file):
    """1) 上传 + 调用后端提取并入库；返回进度、头部/主体 HTML、保存的结构化 JSON + 图像路径列表"""
    if file is None:
        empty_header = "<div class='paper-workbench'><div class='paper-header'><div class='paper-title'>请先上传 PDF 文件</div></div></div>"
        empty_body = "<div class='paper-workbench'><div class='paper-main-grid'></div></div>"
        return "请先上传 PDF 文件", render_progress_html([False] * 5), empty_header, empty_body, {}, []
    steps = [False] * 5
    try:
        # 1. 上传（Gradio 已保存到临时路径）
        steps[0] = True
        progress_html = render_progress_html(steps)

        # 2. 调用后端抽取（可能耗时）
        steps[1] = False
        progress_html = render_progress_html(steps)
        # call backend: extract_paper_structure expects a file path
        structured_json = rag_system.extract_paper_structure(file.name)
        steps[1] = True
        progress_html = render_progress_html(steps)

        # 3. (stage 1 done) 更新索引/embedding
        steps[2] = True  # treat extraction as physics extraction done
        progress_html = render_progress_html(steps)

        # 4. get embedding + persist
        rag_system.update_vector_db(structured_json)
        steps[3] = True
        steps[4] = True
        progress_html = render_progress_html(steps)

        # render cards：顶部摘要 + 底部详细内容 + 图像路径（供 Gallery 使用）
        header_md = render_header_html(structured_json)
        body_md = render_body_html(structured_json)
        fig_paths = extract_figure_paths(structured_json)
        status = "✅ 论文知识提取完成，已存入向量数据库。"
        return status, progress_html, header_md, body_md, structured_json, fig_paths

    except Exception as e:
        # 异常时也必须返回 6 个值，对应：
        # [parse_status, progress_html, paper_header_html, paper_body_html, raw_structured_state, fig_gallery]
        err_status = f"解析失败: {str(e)}"
        progress_html = render_progress_html(steps)
        empty_header = "<div class='paper-workbench'><div class='paper-header'><div class='paper-title'>解析失败</div></div></div>"
        empty_body = "<div class='paper-workbench'><div class='paper-main-grid'></div></div>"
        return err_status, progress_html, empty_header, empty_body, {}, []


def load_demo_case():
    """
    加载内置示例论文（用于前端渲染测试，不调用后端 API）。
    输出签名与 process_pdf_step 一致，便于复用 UI。
    """
    steps = [True] * 5
    progress_html = render_progress_html(steps)
    structured_json = DEMO_STRUCTURED_DATA
    header_md = render_header_html(structured_json)
    body_md = render_body_html(structured_json)
    fig_paths = extract_figure_paths(structured_json)
    status = "✅ 已加载示例论文"
    return status, progress_html, header_md, body_md, structured_json, fig_paths


def render_header_html(data):
    """渲染顶部元数据卡片：标题 + 期刊 + 年份 + 创新点。"""
    if not data or "metadata" not in data:
        return "<div class='paper-workbench'>⚠️ 未能提取到有效数据</div>"

    meta = data.get("metadata", {})
    title = meta.get("title", "未知标题")
    journal = meta.get("journal", "")
    year = meta.get("year", "")
    innovation = meta.get("innovation", "")

    html = []
    html.append("<div class='paper-workbench'>")
    html.append("<div class='paper-header'>")
    html.append(f"<div class='paper-title'>{title}</div>")
    # year 可能是 int，这里统一转成字符串，避免 join 抛出类型错误
    meta_pieces = [journal, year]
    meta_pieces = [str(x) for x in meta_pieces if x not in (None, "")]
    meta_line = " · ".join(meta_pieces)
    if meta_line:
        html.append(f"<div class='paper-meta-line'>{meta_line}</div>")
    if innovation:
        html.append(f"<div class='paper-meta-line'>创新：{innovation}</div>")
    html.append("</div>")  # end header
    html.append("</div>")  # end workbench
    return "".join(html)


def render_body_html(data):
    """渲染底部详细内容：物理背景、现象、参数、力场等（不包含图像本身）。"""
    if not data or "metadata" not in data:
        return "<div class='paper-workbench'>⚠️ 未能提取到有效数据</div>"

    meta = data.get("metadata", {})
    ctx = data.get("physics_context", {})
    params = data.get("parameters", [])
    forces = data.get("force_fields", [])
    figures = data.get("figures", []) or []
    phenomena = data.get("observed_phenomena", "")

    env = ctx.get("environment", "N/A")
    bg = ctx.get("detailed_background", "")

    html = []
    html.append("<div class='paper-workbench'>")
    # 主体两列布局
    html.append("<div class='paper-main-grid'>")

    # 左列：物理背景 + 关键图表 + 观测现象
    html.append("<div class='paper-main-left'>")
    html.append("<div class='paper-card paper-card-physics'>")
    html.append("<h3>物理背景与环境</h3>")
    html.append(f"<p><strong>环境：</strong>{env}</p>")
    if bg:
        html.append(f"<p>{bg}</p>")
    html.append("</div>")

    # 关键图表的真实图片由 Gradio Gallery 组件负责，这里只保留占位标题（避免重复渲染）
    if figures:
        html.append("<div class='paper-card'>")
        html.append("<h3>关键图表 (Scientific Figures)</h3>")
        html.append("<div class='param-section-sub'>下方 Gallery 中展示从 PDF 自动提取的页面快照或图表。</div>")
        html.append("</div>")

    if phenomena:
        html.append("<div class='phenomena-card'>")
        html.append("<div class='phenomena-title'>Observed Phenomena</div>")
        html.append(f"<div class='phenomena-body'>{phenomena}</div>")
        html.append("</div>")

    html.append("</div>")  # end left

    # 右列：参数 grid + 力场
    html.append("<div class='paper-main-right'>")

    # 参数 grid（按物理属性分组）
    html.append("<div class='paper-card'>")
    html.append(
        "<div class='param-section-header'>"
        "<div class='param-section-title'>提取的关键物理参数</div>"
        "<div class='param-section-sub'>按几何 / 电学 / 无量纲进行分组展示</div>"
        "</div>"
    )

    def _param_category(p):
        name = p.get("name", "")
        unit = p.get("unit", "")
        sym = p.get("symbol", "")
        # 几何相关：直径、长度、间距等
        if any(k in name for k in ["直径", "长度", "间距"]) or any(
                s in sym for s in [" d ", "\\lambda", "\\Delta"]
        ):
            return "几何参数"
        # 电学相关：电压、电荷、频率等
        if any(k in name for k in ["电压", "频率", "电荷", "电场"]) or any(
                s in sym for s in ["U_{pp}", " f ", " Q "]
        ):
            return "电学参数"
        # 无量纲 / 控制参数
        if "无量纲" in unit or any(
                k in name for k in ["马赫", "参数", "Mach", "耦合"]
        ):
            return "无量纲与控制参数"
        # 其余（如数密度等）
        return "其他参数"

    if params:
        # 按类别聚类
        grouped = {}
        for p in params:
            cat = _param_category(p)
            grouped.setdefault(cat, []).append(p)

        for cat_name in ["几何参数", "电学参数", "无量纲与控制参数", "其他参数"]:
            items = grouped.get(cat_name, [])
            if not items:
                continue
            html.append(
                "<div class='param-section-header' style='margin-top:4px;'>"
                f"<div class='param-section-title'>{cat_name}</div>"
                "</div>"
            )
            html.append("<div class='param-grid'>")
            for p in items:
                name = p.get("name", "")
                sym = p.get("symbol", "")
                val = p.get("value", "")
                unit = p.get("unit", "")
                meaning = p.get("meaning", "")

                html.append("<div class='param-card'>")
                if sym:
                    html.append(f"<div class='param-symbol'>{sym}</div>")
                if val:
                    html.append(f"<div class='param-value'>{val}</div>")
                if unit:
                    html.append(f"<div class='param-unit'>{unit}</div>")
                if name:
                    html.append(f"<div class='param-name'>{name}</div>")
                if meaning:
                    html.append(f"<div class='param-meaning'>{meaning}</div>")
                html.append("</div>")
            html.append("</div>")
    else:
        html.append("<div class='param-section-sub'>未提取到参数</div>")
    html.append("</div>")  # end param card

    # 力场 cards
    html.append("<div class='paper-card force-section'>")
    html.append("<h3>相互作用力场</h3>")
    if forces:
        for f in forces:
            name = f.get("name", "")
            formula = f.get("formula", "").strip()
            phys = f.get("physical_significance", "")
            comp = f.get("computational_hint", "")

            html.append("<div class='force-card'>")
            html.append(f"<div class='force-name'>{name}</div>")
            if formula:
                # 保留 LaTeX，在 Markdown + KaTeX 环境下渲染
                html.append(f"<div class='force-formula'>$$ {formula} $$</div>")
            if phys:
                html.append(f"<div class='force-text'>物理本质：{phys}</div>")
            if comp:
                html.append(f"<div class='force-text'>计算建议：{comp}</div>")
            html.append("</div>")
    else:
        html.append("<div class='param-section-sub'>未提取到力场</div>")
    html.append("</div>")  # end force section

    html.append("</div>")  # end right
    html.append("</div>")  # end main grid
    html.append("</div>")  # end workbench

    return "".join(html)


def generate_recommendation_step(structured_data, phenomena, param_df, expert_mode=False):
    """把 Dataframe 转成后端需要的 JSON，调用后端生成推荐，并渲染"""
    if not structured_data:
        return "请先上传并解析论文"

    # Dataframe -> dict
    user_input_params = {"expected_phenomena": phenomena or ""}
    if isinstance(param_df, pd.DataFrame):
        df = param_df.fillna("")
        for _, row in df.iterrows():
            name = str(row["参数名称"]).strip()
            if not name:
                continue
            user_input_params[name] = {
                "value": str(row["目标数值"]).strip(),
                "unit": str(row["单位"]).strip(),
                "description": str(row["物理意义"]).strip()
            }

    try:
        raw_res = rag_system.get_simulation_recommendation(structured_data, user_input_params)
        # 后端有时返回 dict，有时返回 JSON 字符串，甚至可能包在 ```json ... ``` 代码块里
        if isinstance(raw_res, dict):
            parsed = raw_res
        else:
            text = str(raw_res).strip()
            # 去掉 Markdown 代码块包裹
            if "```json" in text:
                text = text.split("```json", 1)[1].split("```", 1)[0].strip()
            elif "```" in text:
                text = text.split("```", 1)[1].split("```", 1)[0].strip()
            parsed = safe_json_load(text)
        if not parsed:
            # return raw text
            return raw_res
        # render as professional dashboard layout
        return format_recommendation_panel_v2(parsed, expert_mode)
    except Exception as e:
        return f"生成推荐失败: {str(e)}"


def add_param_row(df):
    """在 Simulation Setup 参数表中追加一行空白参数"""
    try:
        if isinstance(df, pd.DataFrame):
            new_df = df.copy()
            new_df.loc[len(new_df)] = ["", "", "", ""]
            return new_df
        if isinstance(df, list):
            return df + [["", "", "", ""]]
    except Exception as e:
        print(f"[add_param_row] 追加行失败: {e!r}")
    return df


def remove_param_row(df):
    """在 Simulation Setup 参数表中删除最后一行（至少保留一行）"""
    try:
        if isinstance(df, pd.DataFrame):
            if len(df) <= 1:
                return df
            new_df = df.iloc[:-1].copy()
            return new_df
        if isinstance(df, list):
            if len(df) <= 1:
                return df
            return df[:-1]
    except Exception as e:
        print(f"[remove_param_row] 删除行失败: {e!r}")
    return df


def load_demo_recommendation(expert_mode=False):
    """Simulation Setup 视图用的推荐卡片渲染测试：不调用后端，只渲染 DEMO JSON。"""
    try:
        return format_recommendation_panel_v2(DEMO_RECOMMENDATION_JSON, expert_mode)
    except Exception as e:
        return f"渲染示例推荐失败: {e}"


def format_number_scientific(num):
    """格式化数值，支持科学计数法的美观显示"""
    if isinstance(num, (int, float)):
        # 处理科学计数法格式 (如 1.0e4, 1.5e4)
        if abs(num) >= 1e3 or (abs(num) < 1e-2 and num != 0):
            # 使用科学计数法
            exp = int(f"{num:.2e}".split('e')[1])
            base = float(f"{num:.2e}".split('e')[0])
            if abs(base - 1.0) < 0.01:
                return f"10<sup>{exp}</sup>"
            elif abs(base + 1.0) < 0.01:
                return f"−10<sup>{exp}</sup>"
            else:
                # 格式化基数，去除不必要的尾随零
                base_str = f"{base:.2f}".rstrip('0').rstrip('.')
                return f"{base_str}×10<sup>{exp}</sup>"
        else:
            # 普通数值，保留适当小数位
            if abs(num - int(num)) < 1e-10:
                return str(int(num))
            else:
                # 根据数值大小决定小数位数
                if abs(num) >= 1:
                    return f"{num:.2f}".rstrip('0').rstrip('.')
                else:
                    return f"{num:.4f}".rstrip('0').rstrip('.')
    return str(num)


def format_range_display(range_list):
    """格式化范围显示，支持科学计数法"""
    if not range_list or len(range_list) < 2:
        return "N/A"
    start, end = range_list[0], range_list[1]
    start_str = format_number_scientific(start)
    end_str = format_number_scientific(end)
    return f"[{start_str}, {end_str}]"


def convert_formula_to_latex(formula_text):
    """将力场公式文本转换为 LaTeX 格式，确保所有数学符号正确渲染"""
    if not formula_text:
        return ""

    latex = formula_text

    # 1. 先替换希腊字母（在替换其他符号之前）
    greek_map = {
        'λ': r'\lambda', 'θ': r'\theta', 'α': r'\alpha', 'β': r'\beta',
        'γ': r'\gamma', 'Δ': r'\Delta', 'ε': r'\epsilon', 'π': r'\pi',
        'κ': r'\kappa', 'μ': r'\mu', 'ν': r'\nu', 'ρ': r'\rho',
        'σ': r'\sigma', 'τ': r'\tau', 'φ': r'\phi', 'χ': r'\chi',
        'ψ': r'\psi', 'ω': r'\omega', 'Ω': r'\Omega', 'Φ': r'\Phi',
        'Ψ': r'\Psi', 'Σ': r'\Sigma', 'Π': r'\Pi', 'Γ': r'\Gamma',
        'Λ': r'\Lambda', 'Ξ': r'\Xi', 'Θ': r'\Theta'
    }
    for greek, latex_cmd in greek_map.items():
        latex = latex.replace(greek, latex_cmd)

    # 2. 处理上标（Unicode 上标字符）
    superscript_map = {
        '²': '^{2}', '³': '^{3}', '⁴': '^{4}', '⁵': '^{5}',
        '⁶': '^{6}', '⁷': '^{7}', '⁸': '^{8}', '⁹': '^{9}',
        '¹': '^{1}', '⁰': '^{0}'
    }
    for sup, replacement in superscript_map.items():
        # 匹配字母、数字、右括号、右方括号后的上标
        latex = re.sub(r'([A-Za-z0-9\)\]\\]+)' + re.escape(sup), r'\1' + replacement, latex)

    # 3. 处理下标（在希腊字母替换之后）
    # 匹配 \命令_ 或 字母_ 的模式
    latex = re.sub(r'([A-Za-z\\]+)_([A-Za-z0-9]+)', r'\1_{\2}', latex)

    # 4. 处理数学函数
    # 先处理多个反斜杠的情况（\\\\cos, \\\\cos 等 -> \cos）
    latex = re.sub(r'\\\\+cos', r'\\cos', latex)
    latex = re.sub(r'\\\\+sin', r'\\sin', latex)
    latex = re.sub(r'\\\\+tan', r'\\tan', latex)
    latex = re.sub(r'\\\\+exp', r'\\exp', latex)
    latex = re.sub(r'\\\\+ln', r'\\ln', latex)
    latex = re.sub(r'\\\\+log', r'\\log', latex)

    # 处理未转义的函数（cos -> \cos），但避免替换已经在反斜杠后的
    math_functions_unescaped = {
        r'(?<!\\)\bcos\b': r'\\cos',
        r'(?<!\\)\bsin\b': r'\\sin',
        r'(?<!\\)\btan\b': r'\\tan',
        r'(?<!\\)\bexp\b': r'\\exp',
        r'(?<!\\)\bln\b': r'\\ln',
        r'(?<!\\)\blog\b': r'\\log',
    }
    for pattern, replacement in math_functions_unescaped.items():
        latex = re.sub(pattern, replacement, latex)

    # 5. 处理分数：a/b -> \frac{a}{b}（但保持简单分数如 r/λ 不变，除非是复杂分数）
    # 这里保持 / 格式，因为更简洁，MathJax 会自动处理

    # 6. 处理指数表达式：e^{-r/λ} 或 e^{-r/\lambda}
    # 确保指数中的分数正确
    latex = re.sub(r'e\^\{([^}]+)\}', r'e^{\1}', latex)

    # 7. 处理乘号和点号
    latex = latex.replace('×', r'\times')
    latex = latex.replace('·', r'\cdot')
    latex = latex.replace('•', r'\cdot')

    # 8. 处理关系符号
    latex = latex.replace('≈', r'\approx')
    latex = latex.replace('∝', r'\propto')
    latex = latex.replace('≤', r'\leq')
    latex = latex.replace('≥', r'\geq')
    latex = latex.replace('≠', r'\neq')
    latex = latex.replace('±', r'\pm')
    latex = latex.replace('∓', r'\mp')

    # 9. 处理减号和负号
    latex = latex.replace('−', '-')  # Unicode 减号转为 ASCII 减号

    # 10. 处理括号和分隔符
    # 确保括号匹配，但保持原样（LaTeX 会自动处理）

    # 11. 处理空格（LaTeX 中多个空格会被合并，但保留必要的空格）
    latex = re.sub(r'\s+', ' ', latex)  # 多个空格合并为一个

    # 12. 清理多余的转义（如果有）
    latex = latex.strip()

    return latex


def format_recommendation_panel(res_json, expert_mode=False):
    """渲染推荐报告，包含格式化的参数表格和 LaTeX 力场公式"""
    html = card_css()
    html += """
    <style>
      .param-table { width:100%; border-collapse:collapse; margin:12px 0; }
      .param-table th { background:#f8fafc; padding:12px; text-align:left; border-bottom:2px solid #e2e8f0; font-weight:600; font-size:0.9rem; }
      .param-table td { padding:12px; border-bottom:1px solid #e2e8f0; vertical-align:top; }
      .param-table tr:hover { background:#f8fafc; }
      .param-name { font-weight:600; color:#0f172a; font-size:0.95rem; }
      .param-range { font-family:'Courier New', monospace; color:#1e40af; font-weight:500; font-size:0.9rem; }
      .param-step { font-family:'Courier New', monospace; color:#059669; font-weight:500; }
      .param-unit { color:#7c3aed; font-weight:500; }
      .param-reason { color:#475569; font-size:0.85rem; line-height:1.7; }
      .latex-container { background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:24px; margin:20px 0; text-align:center; }
      .latex-formula { font-size:1.2rem; font-family:serif; margin:12px 0; }
      .force-field-name { font-size:1.15rem; font-weight:600; color:#0f172a; margin-bottom:16px; }
      .force-field-reason { color:#475569; line-height:1.8; margin-top:16px; text-align:left; font-size:0.9rem; }
      .math-inline { display:inline-block; margin:0 2px; }
    </style>
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <script>
      window.MathJax = {
        tex: {
          inlineMath: [['$', '$'], ['\\(', '\\)']],
          displayMath: [['$$', '$$'], ['\\[', '\\]']],
          processEscapes: true,
          processEnvironments: true
        }
      };
    </script>
    """

    html += "<div class='card'><h3>🚀 物理对标模拟推荐</h3>"
    html += "<div class='muted' style='margin-bottom:16px;'>请在使用前检查单位与数值的量纲一致性。</div>"

    # Parameter recommendations table
    html += "<h4 style='margin-top:20px; margin-bottom:12px;'>📊 推荐参数区间</h4>"
    html += "<table class='param-table'>"
    html += "<thead><tr><th style='width:18%'>参数名称</th><th style='width:20%'>数值范围</th><th style='width:12%'>步长</th><th style='width:10%'>单位</th><th style='width:40%'>推荐理由</th></tr></thead>"
    html += "<tbody>"

    for p_name, info in res_json.get("parameter_recommendations", {}).items():
        r = info.get("range", ["N/A", "N/A"])
        step = info.get("step", "N/A")
        unit = info.get("unit", "")
        reason = info.get("reason", "")

        # 格式化范围显示
        range_str = format_range_display(r)

        # 格式化步长
        if step != "N/A" and isinstance(step, (int, float)):
            step_str = format_number_scientific(step)
        else:
            step_str = str(step)

        # 处理推荐理由中的数学表达式，转换为 LaTeX
        # 重要：先转换数学符号，再转义 HTML，避免 $ 被转义
        reason_processed = reason

        # 先转换数学符号为 LaTeX（在转义之前）
        reason_processed = re.sub(r'(\d+)\^(\d+)', r'__MATH_START__\1^{\2}__MATH_END__', reason_processed)  # 10^4
        reason_processed = re.sub(r'([A-Za-z_]+)\^(\d+)', r'__MATH_START__\1^{\2}__MATH_END__', reason_processed)  # Q^2
        reason_processed = re.sub(r'([A-Za-z_]+)_([A-Za-z0-9]+)', r'__MATH_START__\1_{\2}__MATH_END__',
                                  reason_processed)  # λ_D
        reason_processed = re.sub(r'([A-Za-z]+)≈', r'__MATH_START__\1 \\approx__MATH_END__', reason_processed)
        reason_processed = re.sub(r'∝', r'__MATH_START__\\propto__MATH_END__', reason_processed)
        reason_processed = re.sub(r'×', r'__MATH_START__\\times__MATH_END__', reason_processed)
        reason_processed = re.sub(r'λ', r'__MATH_START__\\lambda__MATH_END__', reason_processed)
        reason_processed = re.sub(r'θ', r'__MATH_START__\\theta__MATH_END__', reason_processed)
        reason_processed = re.sub(r'κ', r'__MATH_START__\\kappa__MATH_END__', reason_processed)
        reason_processed = re.sub(r'Δ', r'__MATH_START__\\Delta__MATH_END__', reason_processed)
        reason_processed = re.sub(r'ε', r'__MATH_START__\\epsilon__MATH_END__', reason_processed)
        reason_processed = re.sub(r'π', r'__MATH_START__\\pi__MATH_END__', reason_processed)
        reason_processed = re.sub(r'α', r'__MATH_START__\\alpha__MATH_END__', reason_processed)
        reason_processed = re.sub(r'β', r'__MATH_START__\\beta__MATH_END__', reason_processed)
        reason_processed = re.sub(r'γ', r'__MATH_START__\\gamma__MATH_END__', reason_processed)

        # 转义 HTML 特殊字符（但保留数学标记）
        reason_processed = html_escape.escape(reason_processed)

        # 将数学标记替换为 LaTeX 格式
        reason_processed = reason_processed.replace('__MATH_START__', '$').replace('__MATH_END__', '$')

        html += f"""
        <tr>
            <td class='param-name'>{p_name}</td>
            <td class='param-range'>{range_str}</td>
            <td class='param-step'>{step_str}</td>
            <td class='param-unit'>{unit}</td>
            <td class='param-reason'>{reason_processed}</td>
        </tr>
        """

    html += "</tbody></table>"

    # Force field recommendation with LaTeX rendering
    ff = res_json.get("force_field_recommendation", {})
    html += "<hr style='margin:24px 0; border-color:#e2e8f0;'/>"
    html += "<h4 style='margin-top:20px; margin-bottom:12px;'>🧪 推荐模拟力场模型</h4>"

    reason_text = ff.get('reason', '')
    force_name = ff.get('name', 'N/A')

    html += f"<div class='force-field-name'>{force_name}</div>"

    # 提取力场公式（查找形如 W(r,θ) = ... 的公式）
    # 匹配模式：函数名(参数) = 表达式（直到句号、逗号、分号或换行）
    formula_pattern = r'([A-Za-z_]+\([^)]+\))\s*=\s*([^。，；\n]+?)(?=[。，；\n]|$)'
    formula_match = re.search(formula_pattern, reason_text)

    if formula_match:
        formula_name = formula_match.group(1)  # W(r,θ)
        formula_expr = formula_match.group(2).strip()  # 表达式部分

        # 转换为 LaTeX
        formula_latex = convert_formula_to_latex(f"{formula_name} = {formula_expr}")

        # 修复双反斜杠问题（如 \\cos -> \cos）
        # 使用正则表达式处理多个反斜杠的情况
        formula_latex = re.sub(r'\\\\+cos', r'\\cos', formula_latex)
        formula_latex = re.sub(r'\\\\+sin', r'\\sin', formula_latex)
        formula_latex = re.sub(r'\\\\+tan', r'\\tan', formula_latex)
        formula_latex = re.sub(r'\\\\+exp', r'\\exp', formula_latex)
        formula_latex = re.sub(r'\\\\+ln', r'\\ln', formula_latex)
        formula_latex = re.sub(r'\\\\+log', r'\\log', formula_latex)

        # 使用 \\[ \\] 块级公式格式，交由 Gradio Markdown 的 KaTeX 渲染
        html += f"""
        <div class='latex-container'>
            <div style='font-size:1.0rem; margin-bottom:12px; color:#475569; font-weight:500;'>力场公式：</div>
            <div class='latex-formula' style='font-size:1.2rem; text-align:center; padding:12px;'>\\[{formula_latex}\\]</div>
        </div>
        """

    # 处理推荐理由文本，转换数学表达式为 LaTeX
    # 使用占位符方法避免 HTML 转义影响 LaTeX
    reason_with_latex = reason_text

    # 先转换数学符号为占位符
    reason_with_latex = re.sub(r'(\d+)\^(\d+)', r'__MATH_START__\1^{\2}__MATH_END__', reason_with_latex)
    reason_with_latex = re.sub(r'([A-Za-z_]+)\^(\d+)', r'__MATH_START__\1^{\2}__MATH_END__', reason_with_latex)
    reason_with_latex = re.sub(r'([A-Za-z_]+)_([A-Za-z0-9]+)', r'__MATH_START__\1_{\2}__MATH_END__', reason_with_latex)
    reason_with_latex = re.sub(r'([A-Za-z]+)≈', r'__MATH_START__\1 \\approx__MATH_END__', reason_with_latex)
    reason_with_latex = re.sub(r'∝', r'__MATH_START__\\propto__MATH_END__', reason_with_latex)
    reason_with_latex = re.sub(r'×', r'__MATH_START__\\times__MATH_END__', reason_with_latex)
    reason_with_latex = re.sub(r'λ', r'__MATH_START__\\lambda__MATH_END__', reason_with_latex)
    reason_with_latex = re.sub(r'θ', r'__MATH_START__\\theta__MATH_END__', reason_with_latex)
    reason_with_latex = re.sub(r'κ', r'__MATH_START__\\kappa__MATH_END__', reason_with_latex)
    reason_with_latex = re.sub(r'Δ', r'__MATH_START__\\Delta__MATH_END__', reason_with_latex)
    reason_with_latex = re.sub(r'ε', r'__MATH_START__\\epsilon__MATH_END__', reason_with_latex)
    reason_with_latex = re.sub(r'π', r'__MATH_START__\\pi__MATH_END__', reason_with_latex)
    reason_with_latex = re.sub(r'α', r'__MATH_START__\\alpha__MATH_END__', reason_with_latex)
    reason_with_latex = re.sub(r'β', r'__MATH_START__\\beta__MATH_END__', reason_with_latex)
    reason_with_latex = re.sub(r'γ', r'__MATH_START__\\gamma__MATH_END__', reason_with_latex)

    # 转义 HTML 特殊字符
    reason_with_latex = html_escape.escape(reason_with_latex)

    # 将占位符替换为 LaTeX 格式
    reason_with_latex = reason_with_latex.replace('__MATH_START__', '$').replace('__MATH_END__', '$')

    html += f"<div class='force-field-reason'>{reason_with_latex}</div>"

    # Expert details
    if expert_mode:
        html += "<hr style='margin:24px 0; border-color:#e2e8f0;'/>"
        html += "<h4 style='margin-top:20px; margin-bottom:12px;'>🔎 Expert Details (原始 JSON)</h4>"
        html += "<pre style='font-size:0.85rem; background:#f8fafc; padding:16px; border-radius:8px; overflow:auto; max-height:300px; border:1px solid #e2e8f0;'>"
        html += html_escape.escape(json.dumps(res_json, indent=2, ensure_ascii=False))
        html += "</pre>"

    html += "</div>"
    return html


def format_recommendation_panel_v2(res_json, expert_mode=False):
    """
    新版推荐报告渲染：参数卡片 + 力场卡片仪表盘。
    """

    def to_latex_number(num):
        """将数值转换为适合 LaTeX 的科学计数法或普通数值字符串。"""
        if not isinstance(num, (int, float)):
            return html_escape.escape(str(num))
        if num == 0:
            return "0"
        absn = abs(num)
        if absn >= 1e3 or (absn < 1e-2):
            s = f"{num:.4e}"
            base_str, exp_str = s.split("e")
            exp = int(exp_str)
            base = float(base_str)
            if abs(base - 1.0) < 1e-8:
                return f"10^{{{exp}}}"
            else:
                base_clean = f"{base:.2f}".rstrip("0").rstrip(".")
                return f"{base_clean} \\times 10^{{{exp}}}"
        else:
            s = f"{num:.4f}".rstrip("0").rstrip(".")
            return html_escape.escape(s)

    def format_range_latex(range_list):
        if not isinstance(range_list, (list, tuple)) or len(range_list) < 2:
            return "N/A"
        lo, hi = range_list[0], range_list[1]
        return f"$[{to_latex_number(lo)},\\ {to_latex_number(hi)}]$"

    def format_step_latex(step):
        if isinstance(step, (int, float)):
            return f"${to_latex_number(step)}$"
        return html_escape.escape(str(step))

    def format_reason_with_latex(text):
        """将推荐理由中的常见数学模式转成 LaTeX，同时转义 HTML。"""
        if not text:
            return ""
        s = str(text)
        # 典型幂次 / 下标
        s = re.sub(r'(\d+)\^(\d+)', r'__MATH_START__\1^{\2}__MATH_END__', s)
        s = re.sub(r'([A-Za-z_]+)\^(\d+)', r'__MATH_START__\1^{\2}__MATH_END__', s)
        s = re.sub(r'([A-Za-z_]+)_([A-Za-z0-9]+)', r'__MATH_START__\1_{\2}__MATH_END__', s)
        s = s.replace("×", "__MATH_START__\\times__MATH_END__")
        # 常见希腊字母符号
        greek_map = {
            "λ": "\\lambda",
            "θ": "\\theta",
            "κ": "\\kappa",
            "Δ": "\\Delta",
            "ε": "\\epsilon",
            "π": "\\pi",
            "α": "\\alpha",
            "β": "\\beta",
            "γ": "\\gamma",
        }
        for ch, cmd in greek_map.items():
            s = s.replace(ch, f"__MATH_START__{cmd}__MATH_END__")
        # 先转义 HTML，再恢复数学占位符为 $...$
        s = html_escape.escape(s)
        s = s.replace("__MATH_START__", "$").replace("__MATH_END__", "$")
        return s

    # --- 布局容器（CSS 已在 card_css 中全局注入） ---
    html = "<div class='recom-wrapper'>"

    # --- 参数推荐卡片 ---
    for p_name, info in res_json.get("parameter_recommendations", {}).items():
        unit = info.get("unit", "")
        range_list = info.get("range", [])
        step = info.get("step", "N/A")
        reason = info.get("reason", "")
        # 若后端给出 target，优先；否则退回 value 字段；再否则用占位符
        target_val = info.get("target") or info.get("value") or "—"

        name_html = html_escape.escape(str(p_name))
        unit_html = html_escape.escape(str(unit)) if unit else "—"
        target_html = html_escape.escape(str(target_val)) if target_val is not None else "—"

        range_html = format_range_latex(range_list)
        step_html = format_step_latex(step)
        reason_html = format_reason_with_latex(reason)

        card_lines = [
            "<div class=\"recom-card\">",
            "  <div class=\"recom-header\">",
            f"    <div class=\"recom-name\">{name_html}</div>",
            f"    <div class=\"unit-badge\">{unit_html}</div>",
            "  </div>",
            "  <div class=\"recom-values\">",
            "    <div class=\"value-slot\">",
            "      <div class=\"slot-label\">User Target</div>",
            f"      <div class=\"slot-value\">{target_html}</div>",
            "    </div>",
            "    <div class=\"value-slot\">",
            "      <div class=\"slot-label\">Suggested Range</div>",
            f"      <div class=\"range-display\">{range_html}</div>",
            "      <div class=\"range-track\"><div class=\"range-fill\"></div></div>",
            "    </div>",
            "    <div class=\"value-slot\">",
            "      <div class=\"slot-label\">Resolution</div>",
            f"      <div class=\"slot-value\">{step_html}</div>",
            "    </div>",
            "  </div>",
            f"  <div class=\"reason-box\">{reason_html}</div>",
            "</div>",
        ]
        html += "\n".join(card_lines)

    # --- 推荐力场模型卡片 ---
    ff = res_json.get("force_field_recommendation", {})
    force_name = html_escape.escape(str(ff.get("name", "N/A")))
    reason_text = ff.get("reason", "")
    reason_html = format_reason_with_latex(reason_text)

    force_lines = [
        "<div class=\"recom-force-card\">",
        "  <div class=\"recom-force-title\">🧪 Recommended Physical Model</div>",
        f"  <div class=\"recom-force-name\">{force_name}</div>",
        f"  <div class=\"recom-force-body\">{reason_html}</div>",
        "</div>",
    ]
    html += "\n" + "\n".join(force_lines)

    # 可选：附加 Expert JSON
    if expert_mode:
        expert_json = html_escape.escape(json.dumps(res_json, indent=2, ensure_ascii=False))
        html += f"<div class='recom-expert-json'>{expert_json}</div>"

    html += "</div>"
    return html


# ---- Library functions ----
def list_indexed_papers():
    """读取 SQLite，返回表格（title, year, journal, id）"""
    db = rag_system.db_path
    try:
        with sqlite3.connect(db) as conn:
            cur = conn.cursor()
            cur.execute("SELECT id, title, metadata_json FROM papers ORDER BY id DESC")
            rows = cur.fetchall()
            items = []
            for rid, title, meta in rows:
                md = safe_json_load(meta) or {}
                year = md.get("metadata", {}).get("year", "")
                journal = md.get("metadata", {}).get("journal", "")
                items.append([rid, title, journal, year])
            df = pd.DataFrame(items, columns=["id", "title", "journal", "year"])
            return df
    except Exception as e:
        return pd.DataFrame([], columns=["id", "title", "journal", "year"])


def view_paper_metadata(paper_id):
    """点击 library 的某篇，显示 metadata card"""
    try:
        with sqlite3.connect(rag_system.db_path) as conn:
            cur = conn.cursor()
            cur.execute("SELECT metadata_json FROM papers WHERE id = ?", (int(paper_id),))
            r = cur.fetchone()
            if not r:
                return "找不到该论文", ""
            meta = safe_json_load(r[0]) or {}
            # Library 中仍使用完整卡片视图展示（沿用 body + header 的组合）
            header_html = render_header_html(meta)
            body_html = render_body_html(meta)
            html = header_html + body_html
            return html, json.dumps(meta, indent=2, ensure_ascii=False)
    except Exception as e:
        return f"读取出错: {e}", ""


# ---- Build UI ----
with gr.Blocks(
        title="PlasmaRAG"
) as demo:
    # Top status bar
    with gr.Row():
        gr.Markdown(f"## 🔬 PlasmaRAG (v{__version__})")
        with gr.Column(scale=2):
            status_box = gr.Markdown("", elem_id="system_status")

    # 全局注入 card 样式（通过隐藏的 HTML 组件载入 <style>，不在 Markdown 中打印）
    gr.HTML(card_css())


    # runtime system stats
    def get_sys_stats():
        # small stats: count papers, force fields
        try:
            with sqlite3.connect(rag_system.db_path) as conn:
                cur = conn.cursor()
                cur.execute("SELECT COUNT(*) FROM papers")
                n_papers = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM force_fields")
                n_forces = cur.fetchone()[0]
        except:
            n_papers = 0;
            n_forces = 0
        html = f"**Indexed papers:** {n_papers}  •  **Force fields:** {n_forces}  •  **Embedding paper_web:** text-embedding-v2"
        return html


    status_box.value = get_sys_stats()

    with gr.Row():
        # Sidebar
        with gr.Column(scale=1):
            nav = gr.Radio(["Paper Analysis", "Simulation Setup", "Library"], value="Paper Analysis", label="模块切换")
            # upload zone (drag & drop)
            upload = gr.File(label="上传 PDF (拖拽或选择)", file_types=[".pdf"], interactive=True)
            parse_btn = gr.Button("🚀 分析并入库", variant="primary")
            demo_btn = gr.Button("🧪 加载示例论文（渲染测试）", variant="secondary")
            progress_html = gr.HTML(render_progress_html([False] * 5))

            expert_toggle = gr.Checkbox(label="Expert Mode (显示原始 JSON)", value=False)
            # parse status element (in sidebar) - must be defined before use in click handler
            parse_status = gr.Markdown("*等待上传...*", elem_id="parse_status")

        # Main workspace
        with gr.Column(scale=3):
            # Paper Analysis view
            paper_header_html = gr.Markdown(
                "<div class='muted'>解析结果将在此处显示</div>",
                visible=True,
                latex_delimiters=[
                    {"left": "$", "right": "$", "display": False},
                    {"left": "$$", "right": "$$", "display": True},
                    {"left": r"\[", "right": r"\]", "display": True},
                ],
            )
            # Scientific Figures Gallery（紧跟标题下方）
            fig_gallery = gr.Gallery(
                label="Scientific Figures",
                show_label=False,
                value=[],
                columns=[3],
                rows=[1],
                object_fit="contain",
                height=380,
                visible=True,
            )
            # 详细内容（物理背景 + 参数 + 力场）
            paper_body_html = gr.Markdown(
                "",
                visible=True,
                latex_delimiters=[
                    {"left": "$", "right": "$", "display": False},
                    {"left": "$$", "right": "$$", "display": True},
                    {"left": r"\[", "right": r"\]", "display": True},
                ],
            )
            raw_structured_state = gr.State({})

            # Simulation Setup view
            sim_setup_md = gr.Markdown("### Simulation Setup", visible=False)
            phenomena_input = gr.Textbox(label="期望观察到的物理现象", value="观察到微粒在微重力流场中形成的链状结构",
                                         lines=3, visible=False)
            default_params = [
                ["target_particle_charge", "1.2 * 10^4", "e", "目标微粒电荷"],
                ["time_scale", "200.0", "ms", "总演化时长"],
                ["debye_length_target", "0.6", "mm", "系统德拜屏蔽长度"],
            ]
            with gr.Row():
                param_df = gr.Dataframe(
                    headers=["参数名称", "目标数值", "单位", "物理意义"],
                    value=default_params,
                    row_count="dynamic",
                    column_count=(4, "fixed"),
                    datatype=["str", "str", "str", "str"],
                    label="用户模拟参数表 (可增减行)",
                    visible=False,
                )
                with gr.Column(scale=0.1):
                    add_param_row_btn = gr.Button("➕", variant="secondary", visible=False,
                                                  elem_classes=["param-row-btn"])
                    remove_param_row_btn = gr.Button("➖", variant="secondary", visible=False,
                                                     elem_classes=["param-row-btn"])
            recom_btn = gr.Button("💡 生成对标推荐报告", variant="primary", visible=False)
            demo_recom_btn = gr.Button("🧪 加载示例推荐（渲染测试）", variant="secondary", visible=False)
            recom_panel = gr.Markdown(
                "<div class='muted'>推荐结果将在此处显示</div>",
                visible=False,
                latex_delimiters=[
                    {"left": "$", "right": "$", "display": False},
                    {"left": "$$", "right": "$$", "display": True},
                    {"left": r"\[", "right": r"\]", "display": True},
                ],
            )

            # Library view
            lib_md = gr.Markdown("### Indexed Papers", visible=False)
            lib_table = gr.Dataframe(value=list_indexed_papers(), interactive=False, label="已入库论文", visible=False)
            lib_view_btn = gr.Button("📖 查看选中论文", visible=False)
            lib_details_html = gr.HTML("<div class='muted'>论文元数据 / 阅读器</div>", visible=False)
            # State for storing paper metadata JSON
            lib_metadata_state = gr.State("")
            # State: 当前在 Library 里用户选中的论文 ID
            lib_selected_id = gr.State(None)

    # Bind events
    parse_btn.click(fn=process_pdf_step, inputs=[upload],
                    outputs=[parse_status, progress_html, paper_header_html, paper_body_html, raw_structured_state,
                             fig_gallery])
    demo_btn.click(fn=load_demo_case, inputs=[],
                   outputs=[parse_status, progress_html, paper_header_html, paper_body_html, raw_structured_state,
                            fig_gallery])
    # generate / demo recommendation
    recom_btn.click(fn=generate_recommendation_step,
                    inputs=[raw_structured_state, phenomena_input, param_df, expert_toggle], outputs=[recom_panel])
    demo_recom_btn.click(fn=load_demo_recommendation, inputs=[expert_toggle], outputs=[recom_panel])
    # 参数表增删行
    add_param_row_btn.click(fn=add_param_row, inputs=[param_df], outputs=[param_df])
    remove_param_row_btn.click(fn=remove_param_row, inputs=[param_df], outputs=[param_df])


    # library load
    def refresh_library():
        """刷新论文库列表"""
        df = list_indexed_papers()
        return df


    def on_lib_select(evt: gr.SelectData, df):
        """当用户在 Library 表格中点击某一行时，记录其论文 ID。"""
        try:
            if df is None or df.empty:
                return None
            # evt.index 在 Dataframe 中通常为 (row, col)
            row_idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
            paper_id = int(df.iloc[row_idx]["id"])
            return paper_id
        except Exception as e:
            print(f"[on_lib_select] 解析选中行失败: {e!r}")
            return None


    def view_selected_paper(paper_id):
        """处理查看选中论文的逻辑（基于选中的 paper_id）。"""
        if paper_id is None:
            return "请先在表格中点击选择一篇论文", "", []
        try:
            html, json_str = view_paper_metadata(paper_id)
            try:
                data = json.loads(json_str) if json_str else {}
            except Exception:
                data = {}
            fig_paths = extract_figure_paths(data)
            return html, json_str, fig_paths
        except Exception as e:
            return f"读取出错: {e}", "", []


    # Note: In Gradio 6.5.1, Dataframe components don't有 update() method
    # The initial value is set during component creation
    # To refresh, use a separate refresh button or load event
    # 1) 表格行选中事件：更新当前选中的论文 ID
    lib_table.select(fn=on_lib_select, inputs=[lib_table], outputs=[lib_selected_id])
    # 2) 查看按钮：根据选中的 ID 加载论文详情
    lib_view_btn.click(fn=view_selected_paper, inputs=[lib_selected_id],
                       outputs=[lib_details_html, lib_metadata_state, fig_gallery])


    # nav switching (complete visibility control)
    def switch_view(choice):
        """切换视图，控制所有相关组件的可见性"""
        if choice == "Paper Analysis":
            return (
                gr.update(visible=True),  # paper_header_html
                gr.update(visible=True),  # fig_gallery
                gr.update(visible=True),  # paper_body_html
                gr.update(visible=False),  # sim_setup_md
                gr.update(visible=False),  # phenomena_input
                gr.update(visible=False),  # param_df
                gr.update(visible=False),  # recom_btn
                gr.update(visible=False),  # demo_recom_btn
                gr.update(visible=False),  # add_param_row_btn
                gr.update(visible=False),  # remove_param_row_btn
                gr.update(visible=False),  # recom_panel
                gr.update(visible=False),  # lib_md
                gr.update(visible=False),  # lib_table
                gr.update(visible=False),  # lib_view_btn
                gr.update(visible=False)  # lib_details_html
            )
        elif choice == "Simulation Setup":
            return (
                gr.update(visible=False),  # paper_header_html
                gr.update(visible=False),  # fig_gallery
                gr.update(visible=False),  # paper_body_html
                gr.update(visible=True),  # sim_setup_md
                gr.update(visible=True),  # phenomena_input
                gr.update(visible=True),  # param_df
                gr.update(visible=True),  # recom_btn
                gr.update(visible=True),  # demo_recom_btn
                gr.update(visible=True),  # add_param_row_btn
                gr.update(visible=True),  # remove_param_row_btn
                gr.update(visible=True),  # recom_panel
                gr.update(visible=False),  # lib_md
                gr.update(visible=False),  # lib_table
                gr.update(visible=False),  # lib_view_btn
                gr.update(visible=False)  # lib_details_html
            )
        else:  # Library
            # 每次进入 Library 时刷新表格，确保看得到新入库的论文
            refreshed_df = refresh_library()
            return (
                gr.update(visible=False),  # paper_header_html
                gr.update(visible=False),  # fig_gallery
                gr.update(visible=False),  # paper_body_html
                gr.update(visible=False),  # sim_setup_md
                gr.update(visible=False),  # phenomena_input
                gr.update(visible=False),  # param_df
                gr.update(visible=False),  # recom_btn
                gr.update(visible=False),  # demo_recom_btn
                gr.update(visible=False),  # add_param_row_btn
                gr.update(visible=False),  # remove_param_row_btn
                gr.update(visible=False),  # recom_panel
                gr.update(visible=True),  # lib_md
                gr.update(visible=True, value=refreshed_df),  # lib_table
                gr.update(visible=True),  # lib_view_btn
                gr.update(visible=True)  # lib_details_html
            )


    nav.change(
        fn=switch_view,
        inputs=[nav],
        outputs=[
            paper_header_html,
            fig_gallery,
            paper_body_html,
            sim_setup_md,
            phenomena_input,
            param_df,
            recom_btn,
            demo_recom_btn,
            add_param_row_btn,
            remove_param_row_btn,
            recom_panel,
            lib_md,
            lib_table,
            lib_view_btn,
            lib_details_html
        ]
    )

if __name__ == "__main__":
    os.environ["no_proxy"] = "localhost,127.0.0.1"

    # 显式允许前端通过 file= 协议访问项目根目录及图片目录（解决 404 / 沙箱限制）
    current_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(current_dir, "images")
    figures_dir = os.path.join(current_dir, "figures")

    demo.launch(
        theme=gr.themes.Base(
            primary_hue="indigo",
            neutral_hue="slate",
            radius_size="lg"
        ),
        debug=True,
        share=False,
        allowed_paths=[current_dir, images_dir, figures_dir]
    )


