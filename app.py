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
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main .block-container {
        padding-top: 0 !important;
        padding-bottom: 1rem;
        max-width: 95%;
    }
    [data-testid="stAppViewContainer"] > .main {
        padding-top: 0 !important;
    }
    [data-testid="stHeader"] {
        height: 0;
        min-height: 0;
        background: transparent;
    }
    [data-testid="stToolbar"] {
        display: none;
    }
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewContainer"] > .main {
        background: #FAF9F5;
        color: #3D3929;
    }
    [data-testid="stSidebar"] > div:first-child {
        background: #F0EEE6;
        color: #3D3929;
    }
    h1 {
        padding-top: 0 !important;
        margin-top: -0.35rem !important;
        margin-bottom: 0 !important;
        line-height: 1.12;
    }
    h1 + div {
        margin-top: 0 !important;
    }
    .info-box {
        padding: 15px; background: #EFE9DD; color: #3D3929;
        border: 1px solid #DAD5C8; border-left: 5px solid #C15F3C;
        border-radius: 8px; margin-bottom: 20px;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] { border: 1px solid #DAD5C8; background-color: #F0EEE6; border-radius: 8px; padding: 15px; }
    .inactive-box { height: 300px; border: 2px dashed #DAD5C8; border-radius: 8px; display: flex; align-items: center; justify-content: center; color: #6B6553; background-color: #F0EEE6; }
    div[data-testid="stMetricValue"] { font-size: 1.1rem !important; color: #C15F3C; }
    div[data-testid="stMetricLabel"] { font-size: 0.8rem !important; color: #6B6553; }
    [data-testid="stMetric"] { display: flex; flex-direction: column; align-items: center; text-align: center; }
    [data-testid="stMetricValue"] { justify-content: center; font-weight: bold; }
    [data-testid="stMetricLabel"] { justify-content: center; }
    div[data-testid="stAlert"] {
        background: #EFE9DD;
        color: #3D3929;
        border: 1px solid #DAD5C8;
        border-left: 4px solid #C15F3C;
    }
    button:hover {
        border-color: #A84F30 !important;
        color: #A84F30 !important;
    }
    
    /* 优化 Expander 的样式 - 强制加粗 */
    .streamlit-expanderHeader {
        font-size: 1.2em; /* 稍微加大 */
        font-weight: 900 !important; /* 最粗 */
        color: #3D3929;
    }
    .streamlit-expanderHeader p {
        font-weight: 900 !important;
    }
    .course-watermark {
        position: fixed;
        inset: 0;
        z-index: 9999;
        pointer-events: none;
        opacity: 1;
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='360' height='220' viewBox='0 0 360 220'%3E%3Cg transform='translate(180 110) rotate(-28)'%3E%3Ctext x='0' y='0' text-anchor='middle' font-size='28' font-weight='800' fill='rgba(61,57,41,0.06)' font-family='Arial, sans-serif'%3E%E8%A5%BF%E7%94%B5%E9%AB%98%E4%BB%A3%E8%AF%BE%E7%A8%8B%E7%BB%84%3C/text%3E%3C/g%3E%3C/svg%3E");
        background-size: 360px 220px;
        background-position: 0 0, 180px 110px;
    }
</style>
<div class="course-watermark"></div>
""", unsafe_allow_html=True)

# 设备与模型路径配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_LABEL = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
CUDA_LABEL = torch.version.cuda if device.type == "cuda" and torch.version.cuda else "不可用"
MAX_INFERENCE_SIZE = 640
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

    st.markdown("""
    <style>
        [data-testid="stSidebar"] { display: none; }
        .course-watermark { display: none; }
        .main .block-container {
            padding-top: 16vh !important;
            max-width: 1200px;
        }
        [class*="st-key-password_card"] {
            max-width: 400px;
            margin: 0 auto;
            background: #F0EEE6;
            border: 1px solid #DAD5C8 !important;
            border-radius: 8px;
            padding: 2rem !important;
        }
        [class*="st-key-password_input"] button {
            display: none !important;
        }
        [class*="st-key-password_input"] input {
            background: #FAF9F5 !important;
            border: 1px solid #C15F3C !important;
            border-radius: 6px !important;
            padding-right: 0.75rem !important;
        }
        [class*="st-key-password_input"] [data-baseweb="input"] {
            background: #FAF9F5 !important;
            border: 1px solid #C15F3C !important;
            border-radius: 6px !important;
            box-shadow: none !important;
        }
        [class*="st-key-password_input"] [data-testid="InputInstructions"],
        [class*="st-key-password_input"] [data-testid="stTextInputInstructions"],
        [class*="st-key-password_input"] small {
            display: none !important;
        }
        [class*="st-key-password_logo"] [data-testid="stImage"] {
            display: flex;
            justify-content: center;
            width: 100% !important;
        }
        [class*="st-key-password_logo"] {
            align-items: center !important;
        }
        [class*="st-key-password_logo"] img {
            display: block;
            width: 112px !important;
            height: auto !important;
            margin: 0 auto;
        }
    </style>
    """, unsafe_allow_html=True)

    _, password_col, _ = st.columns([1, 1, 1])
    with password_col:
        with st.container(border=True, key="password_card"):
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
# 3. 辅助绘制函数 (可视化轴线与网格)
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

# ==========================================
# 4. 侧边栏 (Sidebar)
# ==========================================

with st.sidebar:
    st.header("🎛️ 矩阵工作台面")
    uploaded_file = st.file_uploader("📂 上传图片（推荐使用肖像照）", type=["jpg", "png", "jpeg"])

    example_dir = "./example"
    example_paths = []
    if os.path.isdir(example_dir):
        example_paths = sorted(
            os.path.join(example_dir, name)
            for name in os.listdir(example_dir)
            if name.lower().endswith((".jpg", ".jpeg", ".png"))
        )

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

    st.divider()
    st.header("🎛️ 系统设置")
    style_opt = st.selectbox("选择动漫风格模型", list(STYLE_MAP.keys()))
    
    st.divider()
    
    st.header("🧠 矩阵编辑器")
    st.info("支持多区域图层叠加编辑")

    # --- 模块 0: 几何变换 ---
    edit_geom = st.checkbox("启用：几何变换 (仿射/透视)", value=False)
    geom_params = {}
    if edit_geom:
        with st.expander("📐 几何与空间变换参数", expanded=True):
            # 1. 平面仿射
            st.markdown("**1. 平面仿射 (2D Affine)**")
            
            # 修复：将 help 文案移动到 slider 的 help 参数中，避免使用 st.help()
            affine_help_text = "基于仿射矩阵 (Affine Matrix) 实现图像的旋转、缩放和平移。\n\n矩阵形式：\n[ [cosθ, -sinθ, tx], \n  [sinθ,  cosθ, ty] ]"
            
            col_g1, col_g2 = st.columns(2)
            with col_g1:
                geom_params['angle'] = st.slider("平面旋转 (Z轴)", -45, 45, 0, help=affine_help_text)
                geom_params['scale'] = st.slider("缩放比例", 0.5, 1.5, 1.0)
            with col_g2:
                geom_params['translate_x'] = st.slider("X轴平移", -100, 100, 0)
                geom_params['translate_y'] = st.slider("Y轴平移", -100, 100, 0)
            
            st.divider()
            
            # 2. 透视投影
            st.markdown("**2. 透视投影 (Perspective)**")
            
            # 修复：将 help 文案移动到 checkbox
            persp_help_text = "基于单应性矩阵 (Homography Matrix) 模拟 3D 空间中的景深效果。\n\n通过改变图像四个角点的映射位置，实现近大远小的视觉透视。"
            
            geom_params['show_grid'] = st.checkbox("显示辅助网格 (Grid)", value=True, help=persp_help_text + "\n\n开启此项可在原图上叠加网格线，以便观察变形。")
            
            geom_params['persp_distortion'] = st.slider("透视强度 (Distortion)", 0.0, 0.5, 0.0, step=0.01, help="控制透视变形的剧烈程度。数值越大，图像边缘收缩越明显。")
            
            direction_map = {"向左倾斜 (Left)": "left", "向右倾斜 (Right)": "right", "向上倾斜 (Top)": "top", "向下倾斜 (Bottom)": "bottom"}
            direction_key = st.selectbox("倾斜方向", list(direction_map.keys()))
            geom_params['persp_direction'] = direction_map[direction_key]
            
            st.divider()
            
            # 3. 镜像反射
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
    
    # --- 模块 1: 头发编辑 ---
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

    # --- 模块 2: 面部编辑 ---
    edit_face = st.checkbox("启用：面部矩阵编辑", value=True)
    face_params = {}
    if edit_face:
        with st.expander("☺️ 面部参数调节", expanded=False):
            face_params['intensity'] = st.slider("腮红强度", 0.0, 2.0, 1.0, key='f_int')
            face_params['color'] = "#FF0000"
    
    st.markdown("---")
    run_analysis = st.button("开始计算 / 更新结果", type="primary", use_container_width=True)


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
    - 调整参数不会自动推理。
    - 点击“开始计算 / 更新结果”后，系统才执行分割、矩阵编辑、几何变换和风格生成。
    - 系统会保留最近两次成功计算，可在工作台中查看新旧结果和参数差异。
    - 页面切换不会清空计算结果、示例图选择或参数状态。
    """)


def render_app_header():
    st.title("西电高等代数实验室")
    st.caption("矩阵分析工作台 · 基于协方差对齐与张量变形的语义风格迁移系统")

    torch_ver = torch.__version__
    st.markdown(f"""
        <style>
            .badge {{ padding: 4px 8px; border-radius: 4px; border: 1px solid; background: #F0EEE6; margin-right: 10px; font-family: monospace; font-size: 0.9em; color: #3D3929; display: inline-block; margin-bottom: 5px; }}
        </style>
        <div>
            <span class="badge" style="border-color: #C15F3C;">⚡ <b style="color:#C15F3C">计算设备:</b> {DEVICE_LABEL}</span>
            <span class="badge" style="border-color: #B4443C;">🔥 <b style="color:#B4443C">Torch版本:</b> v{torch_ver}</span>
            <span class="badge" style="border-color: #5A8A5C;">🚀 <b style="color:#5A8A5C">CUDA环境:</b> {CUDA_LABEL}</span>
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
st.divider()
if current_page == "principle":
    render_principle_page()
    st.stop()
if current_page == "manual":
    render_manual_page()
    st.stop()

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
    st.sidebar.caption("ℹ️ 当前正在展示默认示例图片")

if image is not None and run_analysis:
    image.thumbnail((MAX_INFERENCE_SIZE, MAX_INFERENCE_SIZE))
    
    with st.spinner("正在加载计算图与权重..."):
        parser_net, anime_net = load_resources(style_opt)
        
    if not parser_net or not anime_net: st.stop()

    img_tensor = to_tensor(image).unsqueeze(0).to(device)
    debug_history = {} 

    # --- 核心计算流程 ---
    with st.spinner("正在执行矩阵运算与风格推理..."):
        
        # 1. 基础矩阵编辑 (在原图上进行，用于生成热力图分析)
        # 【缓存机制】：这个 tensor 是不含任何几何变换的，专门用于下方的矩阵分析
        edited_tensor_origin = img_tensor.clone()
        
        # Hair Layer
        if edit_hair:
            mask_hair = get_segmentation_mask(image, parser_net, device, 'hair')
            if mask_hair.sum() > 0:
                edited_tensor_origin, dbg = apply_matrix_color_edit(edited_tensor_origin, mask_hair, hair_params['color'], hair_params['intensity'], 'hair')
                debug_history['hair'] = dbg 

        # Face Layer
        if edit_face:
            mask_face = get_segmentation_mask(image, parser_net, device, 'face')
            if mask_face.sum() > 0:
                edited_tensor_origin, dbg = apply_matrix_color_edit(edited_tensor_origin, mask_face, face_params['color'], face_params['intensity'], 'face')
                debug_history['face'] = dbg 
        
        # 2. 准备几何变换的数据流
        clean_input = img_tensor.clone()
        clean_edited = edited_tensor_origin.clone()
        
        vis_input = img_tensor.clone()
        vis_edited = edited_tensor_origin.clone()

        # 如果开启网格，先在展示流 (Stream B) 上画网格
        if edit_geom and geom_params.get('show_grid', False):
             vis_input = draw_grid_on_tensor(vis_input, color=(200, 200, 200))
             vis_edited = draw_grid_on_tensor(vis_edited, color=(200, 200, 200))

        # 定义变换函数 (支持透视变换)
        def apply_transform(tensor, params):
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
                
                if direction == 'left': endpoints = [[d, d], [w - 1, 0], [w - 1, h - 1], [d, h - 1 - d]]
                elif direction == 'right': endpoints = [[0, 0], [w - 1 - d, d], [w - 1 - d, h - 1 - d], [0, h - 1]]
                elif direction == 'top': endpoints = [[d, d], [w - 1 - d, d], [w - 1, h - 1], [0, h - 1]]
                elif direction == 'bottom': endpoints = [[0, 0], [w - 1, 0], [w - 1, h - 1], [d, h - 1 - d]]
                else: endpoints = startpoints

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

        if edit_geom:
            # 分别对 纯净流 和 展示流 应用相同的几何变换
            clean_edited = apply_transform(clean_edited, geom_params) # 给GAN用的
            
            vis_input = apply_transform(vis_input, geom_params)       # 给Stage1展示用的
            vis_edited = apply_transform(vis_edited, geom_params)     # 给Stage2展示用的
        
        # 3. 图像格式转换 (用于显示)
        vis_input_pil = to_pil_image(vis_input.squeeze(0).clamp(0, 1))
        vis_edited_pil = to_pil_image(vis_edited.squeeze(0).clamp(0, 1))
        
        # 5. GAN 推理 (使用纯净的 clean_edited)
        input_gan = clean_edited * 2 - 1
        with torch.inference_mode():
            out_gan = anime_net(input_gan, align_corners=False)
            out_gan = out_gan.squeeze(0).clip(-1, 1) * 0.5 + 0.5
            vis_anime_pil = to_pil_image(out_gan)

    parameter_snapshot = {
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
        "params": parameter_snapshot,
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


def create_analysis_plot(tensor_mask, tensor_channel, title_prefix):
    data_mask = tensor_mask.squeeze().detach().cpu().numpy()
    data_channel = tensor_channel.squeeze().detach().cpu().numpy()
    flat_data = data_channel.flatten()
    flat_data = flat_data[flat_data > 0.05]
    title_font = {"fontproperties": font_prop} if font_prop else {}

    fig = plt.figure(figsize=(6, 5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.8], wspace=0.3, hspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor('#FAF9F5')
    im1 = ax1.imshow(data_mask, cmap='YlOrBr')
    ax1.set_title("语义掩膜 ($\\mathbf{M}$)", color='#3D3929', fontsize=9, **title_font)
    ax1.axis('off')
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.ax.set_facecolor('#FAF9F5')
    cbar1.ax.yaxis.set_tick_params(color='#3D3929')
    plt.setp(plt.getp(cbar1.ax.axes, 'yticklabels'), color='#3D3929', fontsize=8)
    cbar1.outline.set_edgecolor('none')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor('#FAF9F5')
    im2 = ax2.imshow(data_channel, cmap='YlOrBr')
    ax2.set_title(f"通道响应 ($\\mathbf{{I}}'_{{{title_prefix}}}$)", color='#3D3929', fontsize=9, **title_font)
    ax2.axis('off')
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.ax.set_facecolor('#FAF9F5')
    cbar2.ax.yaxis.set_tick_params(color='#3D3929')
    plt.setp(plt.getp(cbar2.ax.axes, 'yticklabels'), color='#3D3929', fontsize=8)
    cbar2.outline.set_edgecolor('none')

    ax3 = fig.add_subplot(gs[1, :])
    if flat_data.size:
        sns.histplot(flat_data, bins=40, color='#C15F3C', alpha=0.6, kde=True, element="step", fill=True, ax=ax3, line_kws={'linewidth': 1.5})
    ax3.set_title("像素数值分布 / Pixel Value Distribution", color='#3D3929', fontsize=9, pad=10, **title_font)
    ax3.set_facecolor('#FAF9F5')
    ax3.grid(visible=True, which='major', axis='y', color='#DAD5C8', linestyle='--', linewidth=0.5, alpha=0.8)
    ax3.tick_params(axis='both', colors='#3D3929', labelsize=8)
    for label in ax3.get_xticklabels() + ax3.get_yticklabels():
        label.set_color('#3D3929')
    ax3.xaxis.label.set_color('#3D3929')
    ax3.yaxis.label.set_color('#3D3929')
    sns.despine(ax=ax3, left=True, bottom=False)
    ax3.spines['bottom'].set_color('#3D3929')
    ax3.set_ylabel("")
    fig.patch.set_facecolor('#FAF9F5')
    return fig


def render_parameter_summary(params, changed_keys):
    rows = []
    for key, value in params.items():
        safe_key = html.escape(str(key))
        safe_value = html.escape(str(value))
        if key in changed_keys:
            rows.append(f"<div style='color:#B8860B'><b>{safe_key}：</b>{safe_value}（已变化）</div>")
        else:
            rows.append(f"<div style='color:#6B6553'><b>{safe_key}：</b>{safe_value}</div>")
    st.markdown("".join(rows), unsafe_allow_html=True)


last_run = st.session_state["last_run"]
if last_run is not None:
    st.markdown("""
        <style>
        .result-card {
            border: 1px solid #DAD5C8; border-radius: 8px; padding: 16px; background-color: #F0EEE6;
            height: 400px; display: flex; flex-direction: column; justify-content: space-between;
            align-items: center; box-shadow: 0 4px 6px rgba(61,57,41,0.12);
        }
        .result-title { text-align: center; font-weight: 600; font-size: 1em; color: #C15F3C; margin-bottom: 10px; }
        .result-card img { max-height: 280px; width: auto; max-width: 100%; object-fit: contain; border-radius: 4px; flex-grow: 1; margin: 5px 0; }
        .img-caption { text-align: center; font-size: 0.85em; color: #6B6553; margin-top: 8px; font-family: monospace; }
        </style>
        """, unsafe_allow_html=True)

    st.subheader("🖼️ 效果预览 (Process Visualization)")
    col_v1, col_v2, col_v3 = st.columns(3, gap="medium")
    result_cards = [
        (col_v1, "阶段一 · 原始输入", last_run["input"], "已应用几何变换"),
        (col_v2, "阶段二 · 矩阵编辑状态", last_run["edited"], "语义色彩矩阵运算"),
        (col_v3, "阶段三 · 最终输出", last_run["output"], f"风格模型: {last_run['params']['风格模型']}"),
    ]
    for result_col, title, result_image, caption in result_cards:
        with result_col:
            st.markdown(
                f"<div class='result-card'><div class='result-title'>{title}</div>"
                f"<img src='{pil_to_base64(result_image)}'><div class='img-caption'>{caption}</div></div>",
                unsafe_allow_html=True,
            )

    debug_history = last_run["debug_history"]
    st.subheader("📊 矩阵数值分析 (Matrix Analytics Breakdown)")
    if not debug_history:
        st.info("ℹ️ 暂无数据。请在侧边栏勾选“头发”或“面部”编辑以激活矩阵分析模块。")
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

    prev_run = st.session_state["prev_run"]
    if prev_run is not None:
        st.subheader("🔍 新旧结果对比")
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

elif image is not None:
    image.thumbnail((MAX_INFERENCE_SIZE, MAX_INFERENCE_SIZE))
    st.info("当前显示的是预览图片。可以在侧边栏上传自己的图片，调整参数后点击“开始计算 / 更新结果”。")
    preview_left, preview_center, preview_right = st.columns([1.35, 0.9, 1.35])
    with preview_center:
        st.image(image, caption="当前输入预览", width=360)

else:
    st.info("👈 请从侧边栏上传图片以开始。")
