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
from io import BytesIO
import math

# ================= 🔧 字体与环境配置 =================
FONT_PATH = "/home/dengzhao/data/fonts/SimHei/SimHei.ttf"

try:
    if os.path.exists(FONT_PATH):
        fm.fontManager.addfont(FONT_PATH)
        font_prop = fm.FontProperties(fname=FONT_PATH)
        plt.rcParams['font.family'] = font_prop.get_name()
        plt.rcParams['axes.unicode_minus'] = False 
    else:
        print(f"⚠️ 未找到字体文件: {FONT_PATH}")
except Exception as e:
    print(f"⚠️ 字体配置出错: {e}")

from animegan.model import Generator as AnimeGenerator
from faceparsing.model import BiSeNet
from matrix import apply_matrix_color_edit, get_segmentation_mask

# ==========================================
# 1. 页面整体配置与 CSS
# ==========================================

st.set_page_config(
    page_title="矩阵分析与应用",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main .block-container { padding-top: 2rem; max-width: 95%; }
    .info-box { padding: 15px; background-color: #1E1E1E; border-radius: 10px; border-left: 5px solid #00AAFF; margin-bottom: 20px; }
    div[data-testid="stVerticalBlockBorderWrapper"] { border: 1px solid #333; background-color: #161920; border-radius: 8px; padding: 15px; }
    .inactive-box { height: 300px; border: 2px dashed #333; border-radius: 8px; display: flex; align-items: center; justify-content: center; color: #555; background-color: #0E1117; }
    div[data-testid="stMetricValue"] { font-size: 1.1rem !important; color: #00AAFF; }
    div[data-testid="stMetricLabel"] { font-size: 0.8rem !important; color: #888; }
    [data-testid="stMetric"] { display: flex; flex-direction: column; align-items: center; text-align: center; }
    [data-testid="stMetricValue"] { justify-content: center; font-weight: bold; }
    [data-testid="stMetricLabel"] { justify-content: center; }
    
    /* 优化 Expander 的样式 - 强制加粗 */
    .streamlit-expanderHeader {
        font-size: 1.2em; /* 稍微加大 */
        font-weight: 900 !important; /* 最粗 */
        color: #E0E0E0;
    }
    .streamlit-expanderHeader p {
        font-weight: 900 !important;
    }
</style>
""", unsafe_allow_html=True)

# 设备与模型路径配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PARSING_CKPT = "faceparsing/79999_iter.pth"
STYLE_MAP = {
    "风格1:新海诚风": "animegan/weights/face_paint_512_v1.pt",
    "风格2:人像绘风": "animegan/weights/face_paint_512_v2.pt",
    "风格3:宫崎骏风": "animegan/weights/celeba_distill.pt",
    "风格4:红辣椒": "animegan/weights/paprika.pt",
}

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
    st.markdown("""
    <div class="info-box">
        <h3 style="margin-top:0; color:#00AAFF">🎓 课程设计项目</h3>
        <p style="color:#00AAFF"><b>课程名称:</b> 矩阵分析与计算（X2MS1012） </p>
        <p style="color:#00AAFF"><b>指导老师:</b> 尹小艳 </p>
        <p style="color:#00AAFF"><b>论文题目:</b> 《基于高斯分布矩阵与色彩空间线性变换的语义可控动漫生成》 </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
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
            
            # 修复：将 help 文案移动到 checkbox
            reflect_help_text = "模拟镜面反射效果。\n- 水平翻转：绕Y轴翻转，矩阵 x' = -x\n- 垂直翻转：绕X轴翻转，矩阵 y' = -y"
            
            col_r1, col_r2 = st.columns(2)
            with col_r1: geom_params['flip_x'] = st.checkbox("水平翻转 (X-Mirror)", False, help=reflect_help_text)
            with col_r2: geom_params['flip_y'] = st.checkbox("垂直翻转 (Y-Mirror)", False)
    
    # --- 模块 1: 头发编辑 ---
    edit_hair = st.checkbox("启用：头发矩阵编辑", value=True)
    hair_params = {}
    if edit_hair:
        with st.expander("💇‍♀️ 头发参数调节", expanded=False):
            hair_params['color'] = st.color_picker("基础色调", "#a3ff00")
            hair_params['intensity'] = st.slider("处理强度", 0.0, 1.5, 1.0, key='h_int')

    # --- 模块 2: 面部编辑 ---
    edit_face = st.checkbox("启用：面部矩阵编辑", value=True)
    face_params = {}
    if edit_face:
        with st.expander("☺️ 面部参数调节", expanded=False):
            face_params['intensity'] = st.slider("腮红强度", 0.0, 2.0, 1.0, key='f_int')
            face_params['color'] = "#FF0000"
    
    st.markdown("---")
    uploaded_file = st.file_uploader("📂 上传图片 (推荐使用肖像照)", type=["jpg", "png", "jpeg"])

# ==========================================
# 5. 主界面逻辑 (Main Area)
# ==========================================

st.title("🎨 矩阵分析工作台")
st.caption("基于协方差对齐与张量变形的语义风格迁移系统")

torch_ver = torch.__version__
cuda_ver = torch.version.cuda if torch.cuda.is_available() else "12.4"
st.markdown(f"""
    <style>
        .badge {{ padding: 4px 8px; border-radius: 4px; border: 1px solid; background: #1E1E1E; margin-right: 10px; font-family: monospace; font-size: 0.9em; color: #FFF; display: inline-block; margin-bottom: 5px; }}
    </style>
    <div>
        <span class="badge" style="border-color: #00AAFF;">⚡ <b style="color:#00AAFF">计算设备:</b> RTX 4090 (24G) x3</span>
        <span class="badge" style="border-color: #FF4B4B;">🔥 <b style="color:#FF4B4B">Torch版本:</b> v{torch_ver}</span>
        <span class="badge" style="border-color: #00CC00;">🚀 <b style="color:#00CC00">CUDA环境:</b> v{cuda_ver}</span>
    </div>
    """, unsafe_allow_html=True)

# === 新增：平台使用说明书 (加粗标题) ===
with st.expander("📖 平台使用说明书 (User Manual)", expanded=False):
    st.markdown("""
    ### 🛠️ 操作流程指南
    
    1. **📸 图片输入**
       - 侧边栏底部上传肖像照，或直接使用系统加载的默认示例图。
    
    2. **🎨 矩阵编辑 (Matrix Editing)**
       - 在侧边栏开启 `头发` 或 `面部` 编辑。
       - 展开参数面板，通过滑动条调整色彩矩阵的特征值（颜色与强度）。
       - 所有的语义分割与颜色变换都是基于矩阵运算实时生成的。
    
    3. **📐 几何变换 (Geometric Transforms)**
       - 勾选 `启用：几何变换`。
       - **平面仿射**：调整旋转角度、缩放比例和平移。
       - **透视投影**：模拟 3D 空间感，通过调整“透视强度”和“方向”实现近大远小的效果。
       - **镜像反射**：支持水平和垂直翻转。
       - **💡 提示**：鼠标悬停在侧边栏对应功能的小问号上，可查看具体矩阵原理。
       - **注意**：几何变换仅改变视觉展示，下方的矩阵数值分析始终基于原始视角，以保证数据稳定性。
    
    4. **🎭 风格迁移 (Final Generation)**
       - 选择一种动漫风格（如新海诚、宫崎骏）。
       - 系统会将经过矩阵编辑和几何变换后的图像输入 GAN 网络，生成最终画作。
    """)

image = None
DEFAULT_IMAGE_PATH = "./example/test.png" 

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
elif os.path.exists(DEFAULT_IMAGE_PATH):
    image = Image.open(DEFAULT_IMAGE_PATH).convert("RGB")
    st.sidebar.caption("ℹ️ 当前正在展示默认示例图片")

if image is not None:
    image.thumbnail((1024, 1024))
    
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
            # A. 反射
            if params.get('flip_x'): tensor = hflip(tensor)
            if params.get('flip_y'): tensor = vflip(tensor)
            
            # B. 透视变换
            distortion = params.get('persp_distortion', 0.0)
            if distortion > 0:
                _, _, h, w = tensor.shape
                startpoints = [[0, 0], [w, 0], [w, h], [0, h]]
                d = int(distortion * min(w, h))
                direction = params.get('persp_direction', 'left')
                
                if direction == 'left': endpoints = [[0 + d, 0 + d], [w, 0], [w, h], [0 + d, h - d]]
                elif direction == 'right': endpoints = [[0, 0], [w - d, 0 + d], [w - d, h - d], [0, h]]
                elif direction == 'top': endpoints = [[0 + d, 0 + d], [w - d, 0 + d], [w, h], [0, h]]
                elif direction == 'bottom': endpoints = [[0, 0], [w, 0], [w, h], [0 + d, h - d]]
                else: endpoints = startpoints

                tensor = perspective(tensor, startpoints, endpoints, interpolation=InterpolationMode.BILINEAR, fill=0)

            # C. 平面仿射
            tensor = affine(
                tensor, 
                angle=params.get('angle', 0), 
                translate=[params.get('translate_x', 0), params.get('translate_y', 0)], 
                scale=params.get('scale', 1.0), 
                shear=0, interpolation=InterpolationMode.BILINEAR, fill=0
            )
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
        with torch.no_grad():
            out_gan = anime_net(input_gan, align_corners=False)
            out_gan = out_gan.squeeze(0).clip(-1, 1) * 0.5 + 0.5
            vis_anime_pil = to_pil_image(out_gan)

    st.markdown("""
        <style>
        .result-card {
            border: 1px solid #e0e0e0; border-radius: 12px; padding: 16px; background-color: #f4f6f9;
            height: 400px; display: flex; flex-direction: column; justify-content: space-between;
            align-items: center; box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        }
        .result-title { text-align: center; font-weight: 600; font-size: 1em; color: #0077cc; margin-bottom: 10px; }
        .result-card img { max-height: 280px; width: auto; max-width: 100%; object-fit: contain; border-radius: 4px; flex-grow: 1; margin: 5px 0; }
        .img-caption { text-align: center; font-size: 0.85em; color: #555; margin-top: 8px; font-family: monospace; }
        </style>
        """, unsafe_allow_html=True)
        
    st.subheader("🖼️ 效果预览 (Process Visualization)")
    col_v1, col_v2, col_v3 = st.columns(3, gap="medium")

    with col_v1:
        img_b64 = pil_to_base64(vis_input_pil)
        st.markdown(f"""<div class='result-card'><div class='result-title'>阶段一 · 原始输入</div><img src='{img_b64}'><div class='img-caption'>已应用几何变换</div></div>""", unsafe_allow_html=True)

    with col_v2:
        img_b64 = pil_to_base64(vis_edited_pil)
        st.markdown(f"""<div class='result-card'><div class='result-title'>阶段二 · 矩阵编辑状态</div><img src='{img_b64}'><div class='img-caption'>语义色彩矩阵运算</div></div>""", unsafe_allow_html=True)

    with col_v3:
        img_b64 = pil_to_base64(vis_anime_pil)
        st.markdown(f"""<div class='result-card'><div class='result-title'>阶段三 · 最终输出</div><img src='{img_b64}'><div class='img-caption'>风格模型: {style_opt}</div></div>""", unsafe_allow_html=True)

    st.write("")
    st.write("")

    # ==========================================
    # 7. 矩阵数值分析 (静止状态)
    # ==========================================
    st.subheader("📊 矩阵数值分析 (Matrix Analytics Breakdown)")
    if not debug_history:
        st.info("ℹ️ 暂无数据。请在侧边栏勾选“头发”或“面部”编辑以激活矩阵分析模块。")
    else:
        def create_analysis_plot(tensor_mask, tensor_channel, title_prefix):
            data_mask = tensor_mask.squeeze().detach().cpu().numpy()
            data_channel = tensor_channel.squeeze().detach().cpu().numpy()
            flat_data = data_channel.flatten()
            flat_data = flat_data[flat_data > 0.05] 
            
            fig = plt.figure(figsize=(6, 5))
            gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.8], wspace=0.3, hspace=0.35)
            
            # Mask
            ax1 = fig.add_subplot(gs[0, 0])
            im1 = ax1.imshow(data_mask, cmap='magma')
            ax1.set_title(f"语义掩膜 ($\mathbf{{M}}$)", color='white', fontsize=9)
            ax1.axis('off')
            # Colorbar 1
            cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
            cbar1.ax.yaxis.set_tick_params(color='white') 
            plt.setp(plt.getp(cbar1.ax.axes, 'yticklabels'), color='white', fontsize=8) 
            cbar1.outline.set_edgecolor('none') 
            
            # Channel
            ax2 = fig.add_subplot(gs[0, 1])
            im2 = ax2.imshow(data_channel, cmap='viridis')
            ax2.set_title(f"通道响应 ($\mathbf{{I}}'_{{{title_prefix}}}$)", color='white', fontsize=9)
            ax2.axis('off')
            # Colorbar 2
            cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
            cbar2.ax.yaxis.set_tick_params(color='white')
            plt.setp(plt.getp(cbar2.ax.axes, 'yticklabels'), color='white', fontsize=8)
            cbar2.outline.set_edgecolor('none')
            
            # Hist
            ax3 = fig.add_subplot(gs[1, :])
            sns.histplot(flat_data, bins=40, color='#00AAFF', alpha=0.6, kde=True, element="step", fill=True, ax=ax3, line_kws={'linewidth': 1.5})
            ax3.set_title("像素数值分布 (Pixel Value Distribution)", color='white', fontsize=9, pad=10)
            ax3.set_facecolor('#0e1117')
            
            ax3.grid(visible=True, which='major', axis='y', color='#444', linestyle='--', linewidth=0.5, alpha=0.5)
            ax3.tick_params(axis='both', colors='white', labelsize=8) 
            for label in ax3.get_xticklabels() + ax3.get_yticklabels():
                label.set_color('white')
            ax3.xaxis.label.set_color('white')
            ax3.yaxis.label.set_color('white')
            
            sns.despine(ax=ax3, left=True, bottom=False) 
            ax3.spines['bottom'].set_color('#FFFFFF')
            ax3.set_ylabel("") 
            fig.patch.set_facecolor('#161920')
            return fig

        col_ana1, col_ana2 = st.columns(2, gap="large")
        
        with col_ana1:
            st.markdown("<h3 style='text-align: center; margin-bottom: 10px;'>💇‍♀️ 头发矩阵图层</h3>", unsafe_allow_html=True)
            if 'hair' in debug_history:
                with st.container(border=True):
                    d = debug_history['hair']
                    m1, m2, m3 = st.columns(3)
                    val_data = d['Final V'].detach().cpu().numpy().flatten()
                    val_data = val_data[val_data > 0.05]
                    m1.metric("Avg Value", f"{val_data.mean():.2f}")
                    m2.metric("Variance", f"{val_data.var():.3f}")
                    m3.metric("Max Shift", f"{val_data.max():.2f}")
                    st.divider()
                    st.pyplot(create_analysis_plot(d['Processed Mask'], d['Final V'], "v"))
            else:
                st.markdown("<div class='inactive-box'>⚠️ Hair Matrix Inactive</div>", unsafe_allow_html=True)

        with col_ana2:
            st.markdown("<h3 style='text-align: center; margin-bottom: 10px;'>☺️ 面部高斯图层</h3>", unsafe_allow_html=True)
            if 'face' in debug_history:
                with st.container(border=True):
                    d = debug_history['face']
                    m1, m2, m3 = st.columns(3)
                    val_data = d['Final S'].detach().cpu().numpy().flatten()
                    val_data = val_data[val_data > 0.05]
                    m1.metric("Avg Sat.", f"{val_data.mean():.2f}")
                    m2.metric("Variance", f"{val_data.var():.3f}")
                    m3.metric("Peak Int.", f"{val_data.max():.2f}")
                    st.divider()
                    st.pyplot(create_analysis_plot(d['Processed Mask'], d['Final S'], "s"))
            else:
                st.markdown("<div class='inactive-box'>⚠️ Face Matrix Inactive</div>", unsafe_allow_html=True)

else:
    st.info("👈 请从侧边栏上传图片以开始。")

st.divider()

# ==========================================
# 8. 底部：团队贡献
# ==========================================
st.markdown("""
<div style="background-color: #121417; border: 1px solid #00AAFF; padding: 25px; border-radius: 10px; text-align: center;">
    <h3 style="color: #00AAFF; margin-top: 0;">📜 贡献声明</h3>
    <p style="font-size: 1.1em; color: #E0E0E0;">
        本项目由团队全员协作完成。我们在此声明：<b>下列所有成员在理论推导、矩阵算法实现、系统部署及文档编写方面均做出了同等贡献。</b>
    </p>
    <hr style="border: 0; border-top: 1px solid #333; margin: 20px 0;">
    <p style="font-size: 1em; color: #BBBBBB; line-height: 1.8;">
        <b>👥 团队成员</b><br>
        <span style="color: #FFF;"> 唐斌伟 </span> &nbsp;•&nbsp; 
        <span style="color: #FFF;"> 周鑫 </span> &nbsp;•&nbsp; 
        <span style="color: #FFF;"> 梁站 </span> &nbsp;•&nbsp; 
        <span style="color: #FFF;"> 邓钊 </span> &nbsp;•&nbsp; 
        <span style="color: #FFF;"> 宋新杰 </span><br>
        <span style="color: #FFF;"> 田宙 </span> &nbsp;•&nbsp; 
        <span style="color: #FFF;"> 路冰 </span> &nbsp;•&nbsp; 
        <span style="color: #FFF;"> 陈丽汀 </span> &nbsp;•&nbsp; 
        <span style="color: #FFF;"> 彭佳园 </span>
    </p>
</div>
""", unsafe_allow_html=True)