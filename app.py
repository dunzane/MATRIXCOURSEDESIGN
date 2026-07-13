import streamlit as st
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_tensor, to_pil_image, affine, perspective, hflip, vflip, InterpolationMode
import os
import matplotlib.font_manager as fm
import seaborn as sns
import base64
import html
from io import BytesIO

# ==========================================
# 0. 主题令牌（全局唯一的颜色/字体定义处）
# ==========================================
INK = "#201f1d"          # 正文墨色
INK_SOFT = "#6f6a60"     # 次级文字
PAPER = "#f3f2f2"        # 页面底色
PAPER_CARD = "#edebe7"   # 面板底色
HAIRLINE = "#ddd8d0"     # 发丝线
ACCENT = "#b68235"       # 金棕点缀色（只用于描边/文字，不做大面积填充）
ACCENT_DEEP = "#8a5f22"  # 深一档，用于小号文字
SERIF = "'Noto Serif SC', 'Songti SC', 'SimSun', serif"

# ================= 🔧 字体与环境配置 =================
FONT_CANDIDATES = [
    "/home/dengzhao/data/fonts/SimHei/SimHei.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
]
font_prop = None

try:
    for font_path in FONT_CANDIDATES:
        if os.path.exists(font_path):
            fm.fontManager.addfont(font_path)
            font_prop = fm.FontProperties(fname=font_path)
            plt.rcParams['font.family'] = font_prop.get_name()
            break
    plt.rcParams['axes.unicode_minus'] = False
    if font_prop is None:
        print("⚠️ 未找到中文字体，Matplotlib 中文标题可能无法正确显示")
except Exception as e:
    print(f"⚠️ 字体配置出错: {e}")

from animegan.model import Generator as AnimeGenerator
from faceparsing.model import BiSeNet
from matrix import apply_matrix_color_edit, get_segmentation_mask

# ==========================================
# 1. 页面整体配置与 CSS
# ==========================================

st.set_page_config(
    page_title="西电高等代数实验室",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@400;600;700&display=swap');

    /* ---------- 页面骨架 ---------- */
    .main .block-container {{
        padding-top: 0 !important;
        padding-bottom: 2rem;
        max-width: 95%;
    }}
    [data-testid="stAppViewContainer"] > .main {{ padding-top: 0 !important; }}
    [data-testid="stHeader"] {{ height: 0 !important; min-height: 0 !important; background: transparent; }}
    [data-testid="stToolbar"] {{ display: none; }}
    [data-testid="stSidebar"], [data-testid="collapsedControl"] {{ display: none !important; }}
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewContainer"] > .main {{
        background: {PAPER};
        color: {INK};
    }}

    /* ---------- 标题：衬线、书卷气 ---------- */
    h1, h2, h3 {{
        font-family: {SERIF} !important;
        font-weight: 600 !important;
        color: {INK} !important;
        letter-spacing: 0.01em;
    }}
    h1 {{
        padding-top: 0 !important;
        margin-top: -0.55rem !important;
        margin-bottom: -0.12rem !important;
        line-height: 1.08;
    }}
    h1 + div {{ margin-top: 0 !important; }}

    /* ---------- 主区域分节标题 ---------- */
    .sec-head {{
        display: flex; align-items: baseline; gap: 0.7rem;
        border-bottom: 1px solid {HAIRLINE};
        padding-bottom: 0.45rem;
        margin: 1.6rem 0 1rem 0;
    }}
    .sec-num {{
        font-family: {SERIF};
        font-size: 1.5rem; color: {ACCENT};
        font-feature-settings: "tnum";
        line-height: 1;
    }}
    .sec-title {{
        font-family: {SERIF};
        font-size: 1.25rem; font-weight: 600; color: {INK};
        line-height: 1;
    }}
    .sec-note {{ font-size: 0.82rem; color: {INK_SOFT}; margin-left: auto; }}

    /* ---------- 面板与卡片：描边不填充 ---------- */
    div[data-testid="stVerticalBlockBorderWrapper"] {{
        border: 1px solid {HAIRLINE};
        background-color: {PAPER_CARD};
        border-radius: 6px;
        padding: 15px;
    }}
    .inactive-box {{
        height: 300px; border: 1px dashed {HAIRLINE}; border-radius: 6px;
        display: flex; align-items: center; justify-content: center;
        color: {INK_SOFT}; background-color: {PAPER_CARD};
    }}

    div[data-testid="stMetricValue"] {{ font-size: 1.1rem !important; color: {ACCENT_DEEP}; font-feature-settings: "tnum"; }}
    div[data-testid="stMetricLabel"] {{ font-size: 0.8rem !important; color: {INK_SOFT}; }}
    [data-testid="stMetric"] {{ display: flex; flex-direction: column; align-items: center; text-align: center; }}
    [data-testid="stMetricValue"] {{ justify-content: center; font-weight: 600; }}
    [data-testid="stMetricLabel"] {{ justify-content: center; }}

    /* ---------- 顶部导航（分段控件） ---------- */
    [class*="st-key-main_page_nav"] {{ margin-top: 0.1rem !important; margin-bottom: 0 !important; }}
    [class*="st-key-main_page_nav"] [role="radiogroup"] {{
        display: grid !important;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.4rem;
        max-width: 620px;
        margin: 0.1rem 0 0 0;
        padding: 0.25rem;
        border: 1px solid {HAIRLINE};
        border-radius: 6px;
        background: {PAPER_CARD};
    }}
    [class*="st-key-main_page_nav"],
    [class*="st-key-main_page_nav"] > div,
    [class*="st-key-main_page_nav"] [data-testid="stWidgetLabel"],
    [class*="st-key-main_page_nav"] [data-testid="stVerticalBlockBorderWrapper"] {{
        background: transparent !important; border: 0 !important;
        box-shadow: none !important; padding: 0 !important;
    }}
    [class*="st-key-main_page_nav"] label {{
        min-height: 42px;
        padding: 0.5rem 0.8rem !important;
        border: 1px solid transparent !important;
        border-radius: 4px !important;
        background: transparent !important;
        color: {INK} !important;
        display: flex !important; align-items: center !important; justify-content: center !important;
        text-align: center; box-shadow: none !important;
    }}
    [class*="st-key-main_page_nav"] label:has(input:checked) {{
        border-color: {ACCENT} !important;
        background: {PAPER} !important;
    }}
    [class*="st-key-main_page_nav"] label > div:first-child {{ display: none !important; }}
    [class*="st-key-main_page_nav"] label p {{
        font-family: {SERIF} !important;
        font-weight: 600 !important;
        margin: 0 !important; color: {INK} !important; font-size: 0.95rem !important;
    }}
    @media (max-width: 720px) {{
        [class*="st-key-main_page_nav"] [role="radiogroup"] {{ grid-template-columns: 1fr; }}
    }}

    /* ---------- 提示框 ---------- */
    div[data-testid="stAlert"] {{
        background: {PAPER_CARD};
        color: {INK};
        border: 1px solid {HAIRLINE};
        border-left: 3px solid {ACCENT};
        border-radius: 4px;
    }}

    /* ---------- 按钮：描边式，不做实心填充 ---------- */
    .stButton button {{ border-radius: 4px !important; }}
    .stButton button[kind="primary"] {{
        background: transparent !important;
        border: 1px solid {ACCENT} !important;
        color: {ACCENT_DEEP} !important;
        font-weight: 600;
    }}
    .stButton button[kind="primary"]:hover {{
        background: rgba(182, 130, 53, 0.10) !important;
        border-color: {ACCENT_DEEP} !important;
        color: {ACCENT_DEEP} !important;
    }}
    button:hover {{ border-color: {ACCENT_DEEP} !important; color: {ACCENT_DEEP} !important; }}

    /* ---------- 左侧工作面板 ---------- */
    [class*="st-key-workbench_panel"] {{
        background: {PAPER_CARD};
        border: 1px solid {HAIRLINE};
        border-radius: 6px;
        padding: 0.9rem;
    }}
    [class*="st-key-workbench_panel"] h2,
    [class*="st-key-workbench_panel"] h3 {{ margin-top: 0 !important; }}
    [class*="st-key-style_model_select"] [data-baseweb="select"] > div {{
        border: 1px solid {INK} !important;
        border-radius: 4px !important;
        background: {PAPER} !important;
    }}
    [class*="st-key-style_model_select"] [data-baseweb="select"]:focus-within > div {{
        border-color: {INK} !important;
        box-shadow: 0 0 0 1px {INK} !important;
    }}

    /* 面板内的步骤标题 */
    .step-head {{
        display: flex; align-items: center; gap: 0.55rem;
        margin: 0.35rem 0 0.5rem 0;
    }}
    .step-no {{
        width: 22px; height: 22px; flex: 0 0 22px;
        border: 1px solid {ACCENT}; border-radius: 50%;
        color: {ACCENT_DEEP};
        font-size: 0.78rem; font-feature-settings: "tnum";
        display: flex; align-items: center; justify-content: center;
    }}
    .step-title {{
        font-family: {SERIF};
        font-size: 1.02rem; font-weight: 600; color: {INK};
    }}
    .step-sub {{ font-size: 0.75rem; color: {INK_SOFT}; margin-left: auto; }}
    .panel-rule {{ border: 0; border-top: 1px solid {HAIRLINE}; margin: 0.65rem 0 0.55rem 0; }}

    .page-nav-spacer {{ height: 0.15rem; }}

    /* Expander 标题 */
    .streamlit-expanderHeader, .streamlit-expanderHeader p {{
        font-weight: 600 !important;
        color: {INK};
    }}

    /* ---------- 课程组水印（必须保留） ---------- */
    .course-watermark {{
        position: fixed;
        inset: 0;
        z-index: 9999;
        pointer-events: none;
        opacity: 1;
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='360' height='220' viewBox='0 0 360 220'%3E%3Cg transform='translate(180 110) rotate(-28)'%3E%3Ctext x='0' y='0' text-anchor='middle' font-size='28' font-weight='800' fill='rgba(32,31,29,0.055)' font-family='Arial, sans-serif'%3E%E8%A5%BF%E7%94%B5%E9%AB%98%E4%BB%A3%E8%AF%BE%E7%A8%8B%E7%BB%84%3C/text%3E%3C/g%3E%3C/svg%3E");
        background-size: 360px 220px;
        background-position: 0 0, 180px 110px;
    }}
</style>
<div class="course-watermark"></div>
""", unsafe_allow_html=True)

# 设备与模型路径配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_LABEL = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
CUDA_LABEL = torch.version.cuda if device.type == "cuda" and torch.version.cuda else "不可用"
MAX_INFERENCE_SIZE = 640
PREVIEW_SIZE = 420
PARSING_CKPT = "faceparsing/79999_iter.pth"
STYLE_MAP = {
    "风格1:新海诚风": "animegan/weights/face_paint_512_v1.pt",
    "风格2:人像绘风": "animegan/weights/face_paint_512_v2.pt",
    "风格3:宫崎骏风": "animegan/weights/celeba_distill.pt",
    "风格4:红辣椒": "animegan/weights/paprika.pt",
}

APP_PASSWORD = "xdugaodai"

if device.type == "cpu":
    torch.set_num_threads(1)

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False


def require_password():
    if st.session_state["authenticated"]:
        return

    st.markdown(f"""
    <style>
        [data-testid="stSidebar"] {{ display: none; }}
        .course-watermark {{ display: none; }}
        .main .block-container {{
            padding-top: 16vh !important;
            max-width: 1200px;
        }}
        [class*="st-key-password_card"] {{
            max-width: 400px;
            margin: 0 auto;
            background: {PAPER_CARD};
            border: 1px solid {HAIRLINE} !important;
            border-radius: 6px;
            padding: 2rem !important;
        }}
        [class*="st-key-password_input"] button {{ display: none !important; }}
        [class*="st-key-password_input"] input {{
            background: {PAPER} !important;
            border: 0 !important;
            outline: none !important;
            border-radius: 4px !important;
            padding-right: 0.75rem !important;
            box-shadow: none !important;
        }}
        [class*="st-key-password_input"] [data-baseweb="input"] {{
            background: {PAPER} !important;
            border: 1px solid {ACCENT} !important;
            border-radius: 4px !important;
            box-shadow: none !important;
        }}
        [class*="st-key-password_input"] [data-baseweb="input"]:focus-within {{
            border-color: {ACCENT_DEEP} !important;
            box-shadow: 0 0 0 2px rgba(182, 130, 53, 0.18) !important;
        }}
        [class*="st-key-password_input"] [data-testid="InputInstructions"],
        [class*="st-key-password_input"] [data-testid="stTextInputInstructions"],
        [class*="st-key-password_input"] small {{ display: none !important; }}
        [class*="st-key-password_logo"] [data-testid="stImage"] {{
            display: flex; justify-content: center; width: 100% !important;
        }}
        [class*="st-key-password_logo"] {{ align-items: center !important; }}
        [class*="st-key-password_logo"] img {{
            display: block; width: 112px !important; height: auto !important; margin: 0 auto;
        }}
    </style>
    """, unsafe_allow_html=True)

    _, password_col, _ = st.columns([1, 1, 1])
    with password_col:
        with st.container(border=False, key="password_card"):
            with st.container(key="password_logo"):
                st.image("logo/icon.png", width=112, output_format="PNG")
            st.title("西电高等代数实验室")
            st.caption("矩阵分析工作台 · 请输入访问密码后继续")
            password = st.text_input("密码", type="password", key="password_input")
            submitted = st.button("进入实验室", key="password_submit", type="primary", use_container_width=True)

            if submitted:
                if password == APP_PASSWORD:
                    st.session_state["authenticated"] = True
                    st.rerun()
                else:
                    st.error("密码错误，请重新输入。")

    st.stop()

require_password()

for state_key, default_value in {
    "selected_example": None,
    "hair_base_color": "#a3ff00",
    "last_run": None,
    "prev_run": None,
    "page": "workbench",
}.items():
    if state_key not in st.session_state:
        st.session_state[state_key] = default_value


def select_example(path):
    st.session_state["selected_example"] = path


def set_hair_color(color):
    st.session_state["hair_base_color"] = color


def clear_run_history():
    st.session_state["last_run"] = None
    st.session_state["prev_run"] = None


# ==========================================
# 2. 模型加载逻辑
# ==========================================
@st.cache_resource
def load_resources(style_name):
    p_net = BiSeNet(n_classes=19)
    p_net.to(device)
    if os.path.exists(PARSING_CKPT):
        p_net.load_state_dict(torch.load(PARSING_CKPT, map_location=device))
        p_net.eval()

    s_net = AnimeGenerator()
    ckpt = STYLE_MAP.get(style_name)
    if ckpt and os.path.exists(ckpt):
        s_net.load_state_dict(torch.load(ckpt, map_location=device))
        s_net.to(device).eval()
    return p_net, s_net


def pil_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


# ==========================================
# 3. 辅助函数（网格、几何变换、示例图）
# ==========================================

def draw_grid_on_tensor(tensor, step=80, color=(120, 120, 120)):
    """在 Tensor 上绘制网格线，用于观察变形"""
    if tensor.is_cuda:
        tensor_cpu = tensor.cpu()
    else:
        tensor_cpu = tensor

    img_pil = to_pil_image(tensor_cpu.squeeze(0))
    draw = ImageDraw.Draw(img_pil)
    w, h = img_pil.size

    for x in range(0, w, step):
        draw.line([(x, 0), (x, h)], fill=color, width=1)
    for y in range(0, h, step):
        draw.line([(0, y), (w, y)], fill=color, width=1)

    return to_tensor(img_pil).to(tensor.device).unsqueeze(0)


def apply_transform(tensor, params):
    """按固定顺序应用几何变换：仿射 → 透视 → 反射"""
    # A. 平面仿射
    tensor = affine(
        tensor,
        angle=params.get('angle', 0),
        translate=[params.get('translate_x', 0), params.get('translate_y', 0)],
        scale=params.get('scale', 1.0),
        shear=0,
        interpolation=InterpolationMode.BILINEAR,
        fill=0,
    )

    # B. 透视变换（使用仿射变换后的实际尺寸）
    distortion = params.get('persp_distortion', 0.0)
    if distortion > 0:
        _, _, h, w = tensor.shape
        startpoints = [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]
        d = int(distortion * min(w, h))
        direction = params.get('persp_direction', 'left')

        if direction == 'left':
            endpoints = [[d, d], [w - 1, 0], [w - 1, h - 1], [d, h - 1 - d]]
        elif direction == 'right':
            endpoints = [[0, 0], [w - 1 - d, d], [w - 1 - d, h - 1 - d], [0, h - 1]]
        elif direction == 'top':
            endpoints = [[d, d], [w - 1 - d, d], [w - 1, h - 1], [0, h - 1]]
        elif direction == 'bottom':
            endpoints = [[0, 0], [w - 1, 0], [w - 1, h - 1], [d, h - 1 - d]]
        else:
            endpoints = startpoints

        tensor = perspective(tensor, startpoints, endpoints, interpolation=InterpolationMode.BILINEAR, fill=0)

    # C. 反射。对角线反射会交换 H/W，后续步骤读取张量实际尺寸。
    if params.get('flip_x'):
        tensor = hflip(tensor)
    if params.get('flip_y'):
        tensor = vflip(tensor)
    if params.get('reflect_y_eq_x'):
        tensor = torch.transpose(tensor, -1, -2)
    if params.get('reflect_y_eq_neg_x'):
        tensor = torch.transpose(tensor, -1, -2)
        tensor = vflip(hflip(tensor))
    return tensor


def get_example_paths():
    example_dir = "./example"
    if not os.path.isdir(example_dir):
        return []
    return sorted(
        os.path.join(example_dir, name)
        for name in os.listdir(example_dir)
        if name.lower().endswith((".jpg", ".jpeg", ".png"))
    )


def build_snapshot(image_source, style_opt, edit_geom, geom_params, edit_hair, hair_params, edit_face, face_params):
    """把一次运行的全部参数汇总成快照字典（用于展示与新旧对比）"""
    return {
        "图片来源": image_source,
        "风格模型": style_opt,
        "几何变换": "启用" if edit_geom else "关闭",
        "旋转角度": geom_params.get('angle', 0) if edit_geom else "未启用",
        "缩放比例": geom_params.get('scale', 1.0) if edit_geom else "未启用",
        "X轴平移": geom_params.get('translate_x', 0) if edit_geom else "未启用",
        "Y轴平移": geom_params.get('translate_y', 0) if edit_geom else "未启用",
        "透视强度": geom_params.get('persp_distortion', 0.0) if edit_geom else "未启用",
        "透视方向": geom_params.get('persp_direction', "left") if edit_geom else "未启用",
        "x'=-x 反射": bool(geom_params.get('flip_x')) if edit_geom else False,
        "y'=-y 反射": bool(geom_params.get('flip_y')) if edit_geom else False,
        "沿 y=x 反射": bool(geom_params.get('reflect_y_eq_x')) if edit_geom else False,
        "沿 y=-x 反射": bool(geom_params.get('reflect_y_eq_neg_x')) if edit_geom else False,
        "辅助网格": bool(geom_params.get('show_grid')) if edit_geom else False,
        "头发编辑": "启用" if edit_hair else "关闭",
        "头发颜色": hair_params.get('color', "未启用"),
        "头发强度": hair_params.get('intensity', "未启用"),
        "面部编辑": "启用" if edit_face else "关闭",
        "腮红强度": face_params.get('intensity', "未启用"),
    }


def geom_caption(edit_geom, geom_params):
    """把几何参数压缩成一行说明文字，标在结果卡片下方"""
    if not edit_geom:
        return "几何变换：关闭"
    parts = []
    if geom_params.get('angle', 0):
        parts.append(f"旋转 {geom_params['angle']}°")
    if geom_params.get('scale', 1.0) != 1.0:
        parts.append(f"缩放 ×{geom_params['scale']}")
    if geom_params.get('translate_x', 0) or geom_params.get('translate_y', 0):
        parts.append(f"平移 ({geom_params.get('translate_x', 0)}, {geom_params.get('translate_y', 0)})")
    if geom_params.get('persp_distortion', 0.0) > 0:
        parts.append(f"透视 {geom_params['persp_distortion']}")
    flips = []
    if geom_params.get('flip_x'):
        flips.append("x'=-x")
    if geom_params.get('flip_y'):
        flips.append("y'=-y")
    if geom_params.get('reflect_y_eq_x'):
        flips.append("y=x")
    if geom_params.get('reflect_y_eq_neg_x'):
        flips.append("y=-x")
    if flips:
        parts.append("反射 " + "/".join(flips))
    return "几何变换：" + ("、".join(parts) if parts else "启用（参数为默认值）")


def color_caption(edit_hair, hair_params, edit_face, face_params):
    parts = []
    if edit_hair:
        parts.append(f"头发 {hair_params.get('color', '')} ×{hair_params.get('intensity', '')}")
    if edit_face:
        parts.append(f"腮红 ×{face_params.get('intensity', '')}")
    return "语义色彩：" + ("、".join(parts) if parts else "关闭")


def section_head(num, title, note=""):
    st.markdown(
        f"<div class='sec-head'><span class='sec-num'>{num}</span>"
        f"<span class='sec-title'>{html.escape(title)}</span>"
        f"<span class='sec-note'>{html.escape(note)}</span></div>",
        unsafe_allow_html=True,
    )


def step_head(no, title, sub=""):
    st.markdown(
        f"<div class='step-head'><span class='step-no'>{no}</span>"
        f"<span class='step-title'>{html.escape(title)}</span>"
        f"<span class='step-sub'>{html.escape(sub)}</span></div>",
        unsafe_allow_html=True,
    )


def panel_rule():
    st.markdown("<hr class='panel-rule'>", unsafe_allow_html=True)


# ==========================================
# 4. 左侧工作面板（编号步骤）
# ==========================================

def render_workbench_panel(example_paths):
    st.markdown(
        f"<h3 style='font-family:{SERIF}; margin-bottom:0.2rem;'>矩阵工作台面</h3>",
        unsafe_allow_html=True,
    )
    st.caption("从上到下依次完成 4 个步骤，最后点击「开始计算」。")
    panel_rule()

    # ---- 步骤 1 · 图像输入（上传置顶）----
    step_head("1", "图像输入", "上传优先于示例图")
    uploaded_file = st.file_uploader("📂 上传图片（推荐使用肖像照）", type=["jpg", "png", "jpeg"])

    with st.expander("🖼️ 示例图库", expanded=False):
        if example_paths:
            with st.container(height=360, border=False):
                for index, example_path in enumerate(example_paths):
                    preview_col, action_col = st.columns([1, 1])
                    with preview_col:
                        st.image(example_path, width=100)
                    with action_col:
                        is_selected = st.session_state["selected_example"] == example_path
                        st.button(
                            "已选择" if is_selected else "使用此图",
                            key=f"example_{index}",
                            disabled=is_selected,
                            on_click=select_example,
                            args=(example_path,),
                            use_container_width=True,
                        )
                    if index < len(example_paths) - 1:
                        st.divider()
        else:
            st.caption("example 目录中暂无可用图片。")

    panel_rule()

    # ---- 步骤 2 · 几何变换（实时预览）----
    step_head("2", "几何变换", "右侧实时预览")
    edit_geom = st.checkbox("启用：几何变换 (仿射/透视)", value=False)
    geom_params = {}
    if edit_geom:
        with st.expander("📐 几何与空间变换参数", expanded=True):
            st.markdown("**1. 平面仿射 (2D Affine)**")
            affine_help_text = "基于仿射矩阵 (Affine Matrix) 实现图像的旋转、缩放和平移。\n\n矩阵形式：\n[ [cosθ, -sinθ, tx], \n  [sinθ,  cosθ, ty] ]"

            col_g1, col_g2 = st.columns(2)
            with col_g1:
                geom_params['angle'] = st.slider("平面旋转 (Z轴)", -45, 45, 0, help=affine_help_text)
                geom_params['scale'] = st.slider("缩放比例", 0.5, 1.5, 1.0)
            with col_g2:
                geom_params['translate_x'] = st.slider("X轴平移", -100, 100, 0)
                geom_params['translate_y'] = st.slider("Y轴平移", -100, 100, 0)

            st.divider()
            st.markdown("**2. 透视投影 (Perspective)**")
            persp_help_text = "基于单应性矩阵 (Homography Matrix) 模拟 3D 空间中的景深效果。\n\n通过改变图像四个角点的映射位置，实现近大远小的视觉透视。"

            geom_params['show_grid'] = st.checkbox("显示辅助网格 (Grid)", value=True, help=persp_help_text + "\n\n开启此项可在原图上叠加网格线，以便观察变形。")
            geom_params['persp_distortion'] = st.slider("透视强度 (Distortion)", 0.0, 0.5, 0.0, step=0.01, help="控制透视变形的剧烈程度。数值越大，图像边缘收缩越明显。")

            direction_map = {"向左倾斜 (Left)": "left", "向右倾斜 (Right)": "right", "向上倾斜 (Top)": "top", "向下倾斜 (Bottom)": "bottom"}
            direction_key = st.selectbox("倾斜方向", list(direction_map.keys()))
            geom_params['persp_direction'] = direction_map[direction_key]

            st.divider()
            st.markdown("**3. 镜像反射 (Reflection)**")

            reflect_help_text = (
                "四种反射按界面顺序依次叠加。\n\n"
                "- x'=-x：矩阵 [[-1, 0], [0, 1]]\n"
                "- y'=-y：矩阵 [[1, 0], [0, -1]]\n"
                "- 沿 y=x：矩阵 [[0, 1], [1, 0]]\n"
                "- 沿 y=-x：矩阵 [[0, -1], [-1, 0]]"
            )

            col_r1, col_r2 = st.columns(2)
            with col_r1:
                geom_params['flip_x'] = st.checkbox("x' = -x", False, help=reflect_help_text)
                geom_params['reflect_y_eq_x'] = st.checkbox("沿 y = x 反射", False)
            with col_r2:
                geom_params['flip_y'] = st.checkbox("y' = -y", False)
                geom_params['reflect_y_eq_neg_x'] = st.checkbox("沿 y = -x 反射", False)

    panel_rule()

    # ---- 步骤 3 · 语义矩阵编辑 ----
    step_head("3", "语义矩阵编辑", "需点击计算生效")
    st.caption("支持头发 / 面部两个语义图层叠加编辑。")

    edit_hair = st.checkbox("启用：头发矩阵编辑", value=True)
    hair_params = {}
    if edit_hair:
        with st.expander("💇‍♀️ 头发参数调节", expanded=False):
            st.caption("基础色调快捷选择")
            preset_cols = st.columns(3)
            presets = [("棕色", "#8B4513"), ("金黄", "#FFD700"), ("黑色", "#2B2B2B")]
            for preset_col, (label, color) in zip(preset_cols, presets):
                with preset_col:
                    st.button(
                        label,
                        key=f"hair_preset_{color}",
                        on_click=set_hair_color,
                        args=(color,),
                        use_container_width=True,
                    )
            hair_params['color'] = st.color_picker("基础色调", key="hair_base_color")
            hair_params['intensity'] = st.slider("处理强度", 0.0, 1.5, 1.0, key='h_int')

    edit_face = st.checkbox("启用：面部矩阵编辑", value=True)
    face_params = {}
    if edit_face:
        with st.expander("☺️ 面部参数调节", expanded=False):
            face_params['intensity'] = st.slider("腮红强度", 0.0, 2.0, 1.0, key='f_int')
            face_params['color'] = "#FF0000"

    panel_rule()

    # ---- 步骤 4 · 风格模型 ----
    step_head("4", "风格迁移", "计算耗时最长的一步")
    style_opt = st.selectbox("选择动漫风格模型", list(STYLE_MAP.keys()), key="style_model_select")

    panel_rule()
    run_analysis = st.button("▶ 开始计算 / 更新结果", type="primary", use_container_width=True)
    st.caption("几何变换在右侧实时预览；色彩编辑与风格迁移在点击后执行。")

    return uploaded_file, style_opt, edit_geom, geom_params, edit_hair, hair_params, edit_face, face_params, run_analysis


# ==========================================
# 5. 原理介绍 / 使用说明页
# ==========================================

def render_principle_page():
    st.header("📚 原理介绍")
    st.caption("矩阵分析工作台中的线性代数基础")

    st.header("二维仿射变换")
    st.markdown("旋转、缩放和平移可统一写为二维坐标的线性变换与平移组合：")
    st.latex(r"\begin{bmatrix}x'\\y'\end{bmatrix}=\begin{bmatrix}s\cos\theta&-s\sin\theta\\s\sin\theta&s\cos\theta\end{bmatrix}\begin{bmatrix}x\\y\end{bmatrix}+\begin{bmatrix}t_x\\t_y\end{bmatrix}")
    st.latex(r"A=\begin{bmatrix}s\cos\theta&-s\sin\theta&t_x\\s\sin\theta&s\cos\theta&t_y\end{bmatrix}")

    st.header("单应性矩阵与透视投影")
    st.markdown("使用齐次坐标后，平面透视关系可由一个 $3\\times3$ 单应性矩阵表示，最后用 $w'$ 归一化回二维坐标：")
    st.latex(r"\begin{bmatrix}\tilde{x}\\\tilde{y}\\\tilde{w}\end{bmatrix}=H\begin{bmatrix}x\\y\\1\end{bmatrix},\quad H=\begin{bmatrix}h_{11}&h_{12}&h_{13}\\h_{21}&h_{22}&h_{23}\\h_{31}&h_{32}&h_{33}\end{bmatrix}")
    st.latex(r"x'=\frac{\tilde{x}}{\tilde{w}},\qquad y'=\frac{\tilde{y}}{\tilde{w}}")

    st.header("四种镜面反射")
    st.markdown("四种反射都可由二维线性变换矩阵表示：")
    st.latex(r"R_{x'=-x}=\begin{bmatrix}-1&0\\0&1\end{bmatrix},\quad R_{y'=-y}=\begin{bmatrix}1&0\\0&-1\end{bmatrix}")
    st.latex(r"R_{y=x}=\begin{bmatrix}0&1\\1&0\end{bmatrix},\quad R_{y=-x}=\begin{bmatrix}0&-1\\-1&0\end{bmatrix}")

    st.header("语义掩膜与协方差对齐")
    st.markdown("分割网络先生成语义掩膜 $M$，只让目标区域参与色彩变换。理论上的均值平移与协方差缩放可将原分布对齐到目标色彩分布：")
    st.latex(r"\mu=\frac{\sum_i M_i x_i}{\sum_i M_i},\qquad \Sigma=\frac{\sum_i M_i(x_i-\mu)(x_i-\mu)^T}{\sum_i M_i}")
    st.latex(r"x'_i=\mu_t+\Sigma_t^{1/2}\Sigma_s^{-1/2}(x_i-\mu_s),\qquad I'=(1-M)\odot I+M\odot X'")
    st.info("当前 apply_matrix_color_edit 在 HSV 通道上用增益、偏移和软掩膜融合实现这一分布调整思想的轻量近似，并由强度参数控制融合程度。")


def render_manual_page():
    st.header("📖 使用说明")
    st.caption("矩阵分析工作台操作流程")

    st.header("图片输入")
    st.markdown("""
    - 上传图片优先，其次使用选中的示例图，未选择时展示默认图。
    - 示例图库支持直接选择 `example` 目录中的图片。
    """)

    st.header("几何变换")
    st.markdown("""
    - 几何变换是独立视觉模块，可单独启用或关闭。
    - 几何变换的参数会在右侧「实时预览」区即时生效，无需点击计算。
    - 几何变换仅影响预览图与最终生成结果，不参与矩阵数值分析。
    - 处理顺序固定为仿射变换、透视投影、镜面反射。
    """)

    st.header("头发与面部矩阵编辑")
    st.markdown("""
    - 头发矩阵编辑和面部矩阵编辑可单独开启，也可同时开启并叠加效果。
    - 矩阵数值分析只统计这两个语义图层。
    """)

    st.header("计算与结果")
    st.markdown("""
    - 色彩编辑与风格迁移不会自动推理，需点击「开始计算 / 更新结果」。
    - 点击后系统执行分割、矩阵编辑、几何变换和风格生成。
    - 若调整了参数但尚未重新计算，结果区顶部会出现"参数已修改"的提示。
    - 系统会保留最近两次成功计算，可在工作台中查看新旧结果和参数差异。
    - 页面切换不会清空计算结果、示例图选择或参数状态。
    """)


# ==========================================
# 6. 页头与导航
# ==========================================

def render_app_header():
    st.title("西电高等代数实验室")
    st.caption("矩阵分析工作台 · 基于协方差对齐与张量变形的语义风格迁移系统")

    torch_ver = torch.__version__
    st.markdown(f"""
        <style>
            .badge {{ padding: 4px 10px; border-radius: 4px; border: 1px solid {HAIRLINE}; background: {PAPER_CARD}; margin-right: 10px; font-family: ui-monospace, monospace; font-size: 0.85em; color: {INK}; display: inline-block; margin-bottom: 5px; font-feature-settings: "tnum"; }}
            .badge b {{ color: {ACCENT_DEEP}; font-weight: 600; }}
        </style>
        <div>
            <span class="badge">⚡ <b>计算设备:</b> {DEVICE_LABEL}</span>
            <span class="badge">🔥 <b>Torch版本:</b> v{torch_ver}</span>
            <span class="badge">🚀 <b>CUDA环境:</b> {CUDA_LABEL}</span>
        </div>
        """, unsafe_allow_html=True)


render_app_header()

page_navigation = {
    "🎨 矩阵工作台": "workbench",
    "📚 原理介绍": "principle",
    "📖 使用说明": "manual",
}
page_labels = list(page_navigation)
current_page = st.session_state["page"]
current_label = next(
    label for label, page_key in page_navigation.items()
    if page_key == current_page
)
selected_label = st.radio(
    "页面导航",
    page_labels,
    index=page_labels.index(current_label),
    horizontal=True,
    label_visibility="collapsed",
    key="main_page_nav",
)
st.session_state["page"] = page_navigation[selected_label]
current_page = st.session_state["page"]
st.markdown("<div class='page-nav-spacer'></div>", unsafe_allow_html=True)

if current_page == "principle":
    render_principle_page()
    st.stop()
if current_page == "manual":
    render_manual_page()
    st.stop()

# ==========================================
# 7. 工作台主页面
# ==========================================

example_paths = get_example_paths()
workbench_panel_col, workbench_content_col = st.columns([0.24, 0.76], gap="large")
with workbench_panel_col:
    with st.container(border=True, key="workbench_panel"):
        (
            uploaded_file,
            style_opt,
            edit_geom,
            geom_params,
            edit_hair,
            hair_params,
            edit_face,
            face_params,
            run_analysis,
        ) = render_workbench_panel(example_paths)

with workbench_content_col:
    # ---------- 输入图像解析 ----------
    image = None
    image_source = ""

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        image_source = f"上传图片：{uploaded_file.name}"
    elif st.session_state["selected_example"] and os.path.exists(st.session_state["selected_example"]):
        image = Image.open(st.session_state["selected_example"]).convert("RGB")
        image_source = f"示例图片：{os.path.basename(st.session_state['selected_example'])}"
    elif example_paths:
        default_example_path = example_paths[0]
        image = Image.open(default_example_path).convert("RGB")
        image_source = f"默认图片：{os.path.basename(default_example_path)}"

    if image is None:
        st.info("👈 请从左侧工作面板上传图片以开始。")
        st.stop()

    # 当前参数快照（用于过期提示与本次计算记录）
    current_snapshot = build_snapshot(
        image_source, style_opt,
        edit_geom, geom_params,
        edit_hair, hair_params,
        edit_face, face_params,
    )

    # ---------- ① 实时预览（几何变换即时生效，无需推理） ----------
    section_head("01", "输入 · 实时预览", "几何变换即时生效 · 色彩与风格需点击「开始计算」")

    if image_source.startswith("默认"):
        st.caption(f"ℹ️ 当前正在展示默认示例图片（{image_source}），可在左侧上传替换。")
    else:
        st.caption(f"当前输入 · {image_source}")

    preview_img = image.copy()
    preview_img.thumbnail((PREVIEW_SIZE, PREVIEW_SIZE))
    preview_tensor = to_tensor(preview_img).unsqueeze(0)
    if edit_geom and geom_params.get('show_grid', False):
        preview_tensor = draw_grid_on_tensor(preview_tensor, step=60, color=(200, 200, 200))
    if edit_geom:
        preview_tensor = apply_transform(preview_tensor, geom_params)
    preview_transformed = to_pil_image(preview_tensor.squeeze(0).clamp(0, 1))

    live_col1, live_col2, live_col3 = st.columns([1, 1, 1.1])
    with live_col1:
        st.image(preview_img, caption="原始输入", use_container_width=True)
    with live_col2:
        st.image(
            preview_transformed,
            caption=geom_caption(edit_geom, geom_params),
            use_container_width=True,
        )
    with live_col3:
        st.markdown(
            f"""
            <div style="border:1px solid {HAIRLINE}; border-radius:6px; background:{PAPER_CARD}; padding:14px 16px; font-size:0.86rem; color:{INK_SOFT}; line-height:1.8;">
            <b style="color:{INK}; font-family:{SERIF};">待计算的编辑</b><br>
            {html.escape(color_caption(edit_hair, hair_params, edit_face, face_params))}<br>
            风格模型：{html.escape(style_opt)}<br>
            <span style="color:{ACCENT_DEEP};">点击左侧「开始计算 / 更新结果」执行分割、色彩矩阵编辑与风格迁移。</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ---------- 核心计算 ----------
    if run_analysis:
        run_image = image.copy()
        run_image.thumbnail((MAX_INFERENCE_SIZE, MAX_INFERENCE_SIZE))

        with st.spinner("正在加载计算图与权重..."):
            parser_net, anime_net = load_resources(style_opt)

        if not parser_net or not anime_net:
            st.stop()

        img_tensor = to_tensor(run_image).unsqueeze(0).to(device)
        debug_history = {}

        with st.spinner("正在执行矩阵运算与风格推理..."):

            # 1. 基础矩阵编辑（在原图上进行，用于生成热力图分析）
            edited_tensor_origin = img_tensor.clone()

            if edit_hair:
                mask_hair = get_segmentation_mask(run_image, parser_net, device, 'hair')
                if mask_hair.sum() > 0:
                    edited_tensor_origin, dbg = apply_matrix_color_edit(edited_tensor_origin, mask_hair, hair_params['color'], hair_params['intensity'], 'hair')
                    debug_history['hair'] = dbg

            if edit_face:
                mask_face = get_segmentation_mask(run_image, parser_net, device, 'face')
                if mask_face.sum() > 0:
                    edited_tensor_origin, dbg = apply_matrix_color_edit(edited_tensor_origin, mask_face, face_params['color'], face_params['intensity'], 'face')
                    debug_history['face'] = dbg

            # 2. 几何变换的数据流
            clean_edited = edited_tensor_origin.clone()

            vis_input = img_tensor.clone()
            vis_edited = edited_tensor_origin.clone()

            if edit_geom and geom_params.get('show_grid', False):
                vis_input = draw_grid_on_tensor(vis_input, color=(200, 200, 200))
                vis_edited = draw_grid_on_tensor(vis_edited, color=(200, 200, 200))

            if edit_geom:
                clean_edited = apply_transform(clean_edited, geom_params)   # 给 GAN 用
                vis_input = apply_transform(vis_input, geom_params)         # 阶段一展示
                vis_edited = apply_transform(vis_edited, geom_params)       # 阶段二展示

            vis_input_pil = to_pil_image(vis_input.squeeze(0).clamp(0, 1))
            vis_edited_pil = to_pil_image(vis_edited.squeeze(0).clamp(0, 1))

            # 3. GAN 推理（使用纯净的 clean_edited）
            input_gan = clean_edited * 2 - 1
            with torch.inference_mode():
                out_gan = anime_net(input_gan, align_corners=False)
                out_gan = out_gan.squeeze(0).clip(-1, 1) * 0.5 + 0.5
                vis_anime_pil = to_pil_image(out_gan)

        debug_history_cpu = {}
        for layer, layer_data in debug_history.items():
            value_key = "Final V" if layer == "hair" else "Final S"
            debug_history_cpu[layer] = {
                key: layer_data[key].detach().cpu()
                for key in ("Processed Mask", value_key)
                if key in layer_data
            }
        new_run = {
            "input": vis_input_pil.copy(),
            "edited": vis_edited_pil.copy(),
            "output": vis_anime_pil.copy(),
            "params": dict(current_snapshot),
            "debug_history": debug_history_cpu,
        }
        previous_run = st.session_state["last_run"]
        if previous_run is not None:
            previous_run = {
                key: value for key, value in previous_run.items()
                if key != "debug_history"
            }
        st.session_state["prev_run"] = previous_run
        st.session_state["last_run"] = new_run

    # ---------- 分析绘图 ----------
    def create_analysis_plot(tensor_mask, tensor_channel, title_prefix):
        data_mask = tensor_mask.squeeze().detach().cpu().numpy()
        data_channel = tensor_channel.squeeze().detach().cpu().numpy()
        flat_data = data_channel.flatten()
        flat_data = flat_data[flat_data > 0.05]
        title_font = {"fontproperties": font_prop} if font_prop else {}

        fig = plt.figure(figsize=(6, 5))
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.8], wspace=0.3, hspace=0.35)

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_facecolor(PAPER)
        im1 = ax1.imshow(data_mask, cmap='YlOrBr')
        ax1.set_title("语义掩膜 ($\\mathbf{M}$)", color=INK, fontsize=9, **title_font)
        ax1.axis('off')
        cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        cbar1.ax.set_facecolor(PAPER)
        cbar1.ax.yaxis.set_tick_params(color=INK)
        plt.setp(plt.getp(cbar1.ax.axes, 'yticklabels'), color=INK, fontsize=8)
        cbar1.outline.set_edgecolor('none')

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_facecolor(PAPER)
        im2 = ax2.imshow(data_channel, cmap='YlOrBr')
        ax2.set_title(f"通道响应 ($\\mathbf{{I}}'_{{{title_prefix}}}$)", color=INK, fontsize=9, **title_font)
        ax2.axis('off')
        cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        cbar2.ax.set_facecolor(PAPER)
        cbar2.ax.yaxis.set_tick_params(color=INK)
        plt.setp(plt.getp(cbar2.ax.axes, 'yticklabels'), color=INK, fontsize=8)
        cbar2.outline.set_edgecolor('none')

        ax3 = fig.add_subplot(gs[1, :])
        if flat_data.size:
            sns.histplot(flat_data, bins=40, color=ACCENT, alpha=0.6, kde=True, element="step", fill=True, ax=ax3, line_kws={'linewidth': 1.5})
        ax3.set_title("像素数值分布 / Pixel Value Distribution", color=INK, fontsize=9, pad=10, **title_font)
        ax3.set_facecolor(PAPER)
        ax3.grid(visible=True, which='major', axis='y', color=HAIRLINE, linestyle='--', linewidth=0.5, alpha=0.8)
        ax3.tick_params(axis='both', colors=INK, labelsize=8)
        for label in ax3.get_xticklabels() + ax3.get_yticklabels():
            label.set_color(INK)
        ax3.xaxis.label.set_color(INK)
        ax3.yaxis.label.set_color(INK)
        sns.despine(ax=ax3, left=True, bottom=False)
        ax3.spines['bottom'].set_color(INK)
        ax3.set_ylabel("")
        fig.patch.set_facecolor(PAPER)
        return fig

    def render_parameter_summary(params, changed_keys):
        rows = []
        for key, value in params.items():
            safe_key = html.escape(str(key))
            safe_value = html.escape(str(value))
            if key in changed_keys:
                rows.append(f"<div style='color:{ACCENT_DEEP}'><b>{safe_key}：</b>{safe_value}（已变化）</div>")
            else:
                rows.append(f"<div style='color:{INK_SOFT}'><b>{safe_key}：</b>{safe_value}</div>")
        st.markdown("".join(rows), unsafe_allow_html=True)

    # ---------- ② 计算结果 ----------
    last_run = st.session_state["last_run"]
    if last_run is None:
        section_head("02", "计算结果", "尚未计算")
        st.info("调整好参数后，点击左侧「开始计算 / 更新结果」，此处将展示三阶段处理结果与矩阵数值分析。")
    else:
        # 参数过期提示：当前面板参数与结果所用参数不一致时给出醒目提醒
        stale_keys = [
            key for key in current_snapshot
            if current_snapshot.get(key) != last_run["params"].get(key)
        ]
        note = "结果与当前参数一致" if not stale_keys else f"参数已修改（{ '、'.join(stale_keys[:3]) }{'…' if len(stale_keys) > 3 else ''}），结果尚未更新"
        section_head("02", "计算结果", note)
        if stale_keys:
            st.warning("⚠️ 左侧参数在上次计算后被修改，下方结果对应的是**上一次计算的参数**。点击「开始计算 / 更新结果」刷新。")

        st.markdown(f"""
            <style>
            .result-card {{
                border: 1px solid {HAIRLINE}; border-radius: 6px; padding: 16px; background-color: {PAPER_CARD};
                height: 420px; display: flex; flex-direction: column; justify-content: space-between;
                align-items: center;
            }}
            .result-title {{ text-align: center; font-weight: 600; font-size: 1em; color: {INK}; margin-bottom: 8px; font-family: {SERIF}; }}
            .result-step {{ color: {ACCENT}; font-feature-settings: "tnum"; margin-right: 6px; }}
            .result-card img {{ max-height: 280px; width: auto; max-width: 100%; object-fit: contain; border-radius: 3px; flex-grow: 1; margin: 5px 0; border: 1px solid {HAIRLINE}; }}
            .img-caption {{ text-align: center; font-size: 0.8em; color: {INK_SOFT}; margin-top: 8px; font-family: ui-monospace, monospace; line-height: 1.5; }}
            </style>
            """, unsafe_allow_html=True)

        run_params = last_run["params"]
        # 每张卡片下标注"驱动它的参数"，让参数与结果一一对应
        cap1 = geom_caption(run_params.get("几何变换") == "启用", {
            'angle': run_params.get("旋转角度", 0) if run_params.get("旋转角度") != "未启用" else 0,
            'scale': run_params.get("缩放比例", 1.0) if run_params.get("缩放比例") != "未启用" else 1.0,
            'translate_x': run_params.get("X轴平移", 0) if run_params.get("X轴平移") != "未启用" else 0,
            'translate_y': run_params.get("Y轴平移", 0) if run_params.get("Y轴平移") != "未启用" else 0,
            'persp_distortion': run_params.get("透视强度", 0.0) if run_params.get("透视强度") != "未启用" else 0.0,
            'flip_x': run_params.get("x'=-x 反射", False),
            'flip_y': run_params.get("y'=-y 反射", False),
            'reflect_y_eq_x': run_params.get("沿 y=x 反射", False),
            'reflect_y_eq_neg_x': run_params.get("沿 y=-x 反射", False),
        })
        hair_on = run_params.get("头发编辑") == "启用"
        face_on = run_params.get("面部编辑") == "启用"
        cap2_parts = []
        if hair_on:
            cap2_parts.append(f"头发 {run_params.get('头发颜色')} ×{run_params.get('头发强度')}")
        if face_on:
            cap2_parts.append(f"腮红 ×{run_params.get('腮红强度')}")
        cap2 = "语义色彩：" + ("、".join(cap2_parts) if cap2_parts else "关闭")
        cap3 = f"风格模型：{run_params.get('风格模型')}"

        col_v1, col_v2, col_v3 = st.columns(3, gap="medium")
        result_cards = [
            (col_v1, "一", "原始输入 + 几何变换", last_run["input"], cap1),
            (col_v2, "二", "语义矩阵编辑", last_run["edited"], cap2),
            (col_v3, "三", "风格迁移输出", last_run["output"], cap3),
        ]
        for result_col, step_no, title, result_image, caption in result_cards:
            with result_col:
                st.markdown(
                    f"<div class='result-card'><div class='result-title'><span class='result-step'>阶段{step_no}</span>{html.escape(title)}</div>"
                    f"<img src='{pil_to_base64(result_image)}'><div class='img-caption'>{html.escape(caption)}</div></div>",
                    unsafe_allow_html=True,
                )

        with st.expander("📋 本次计算的完整参数快照", expanded=False):
            render_parameter_summary(run_params, [])

        # ---------- ③ 矩阵数值分析 ----------
        debug_history = last_run["debug_history"]
        section_head("03", "矩阵数值分析", "Matrix Analytics Breakdown")
        if not debug_history:
            st.info("ℹ️ 暂无数据。请在左侧工作面板勾选「头发」或「面部」编辑以激活矩阵分析模块。")
        else:
            col_ana1, col_ana2 = st.columns(2, gap="large")
            analysis_configs = [
                (col_ana1, "💇‍♀️ 头发矩阵图层", "hair", "Final V", "v", ("Avg Value", "Variance", "Max Shift")),
                (col_ana2, "☺️ 面部高斯图层", "face", "Final S", "s", ("Avg Sat.", "Variance", "Peak Int.")),
            ]
            for analysis_col, title, layer, value_key, channel, metric_labels in analysis_configs:
                with analysis_col:
                    st.markdown(f"<h3 style='text-align:center; margin-bottom:10px;'>{title}</h3>", unsafe_allow_html=True)
                    if layer not in debug_history:
                        st.markdown(f"<div class='inactive-box'>⚠️ {layer.title()} Matrix Inactive</div>", unsafe_allow_html=True)
                        continue
                    with st.container(border=True):
                        data = debug_history[layer]
                        val_data = data[value_key].detach().cpu().numpy().flatten()
                        val_data = val_data[val_data > 0.05]
                        metric_cols = st.columns(3)
                        metric_values = (
                            f"{val_data.mean():.2f}" if val_data.size else "--",
                            f"{val_data.var():.3f}" if val_data.size else "--",
                            f"{val_data.max():.2f}" if val_data.size else "--",
                        )
                        for metric_col, label, value in zip(metric_cols, metric_labels, metric_values):
                            metric_col.metric(label, value)
                        st.divider()
                        fig = create_analysis_plot(data['Processed Mask'], data[value_key], channel)
                        st.pyplot(fig)
                        plt.close(fig)

        # ---------- ④ 新旧结果对比 ----------
        prev_run = st.session_state["prev_run"]
        if prev_run is not None:
            section_head("04", "新旧结果对比", "参数变化项以金色高亮")
            changed_keys = [
                key for key in last_run["params"]
                if last_run["params"].get(key) != prev_run["params"].get(key)
            ][:2]
            if not changed_keys:
                st.caption("两次运行的参数相同。")
            compare_old, compare_new = st.columns(2, gap="large")
            with compare_old:
                st.markdown("#### 上一次结果")
                st.image(prev_run["output"], use_container_width=True)
                render_parameter_summary(prev_run["params"], changed_keys)
            with compare_new:
                st.markdown("#### 本次结果")
                st.image(last_run["output"], use_container_width=True)
                render_parameter_summary(last_run["params"], changed_keys)
            st.button("清空历史", on_click=clear_run_history)
