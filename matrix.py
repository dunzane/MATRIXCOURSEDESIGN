import cv2
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as transforms

def rgb_to_hsv(image_tensor):
    """
    PyTorch 版 RGB -> HSV 变换 (Differentiable)
    Input: [B, 3, H, W], Range [0, 1]
    Output: H, S, V separated tensors
    """
    r, g, b = image_tensor[:, 0, ...], image_tensor[:, 1, ...], image_tensor[:, 2, ...]
    
    max_val, _ = image_tensor.max(dim=1)
    min_val, _ = image_tensor.min(dim=1)
    diff = max_val - min_val + 1e-6

    # V (Value)
    v = max_val

    # S (Saturation)
    s = diff / (max_val + 1e-6)
    s[max_val == 0] = 0

    # H (Hue)
    h = torch.zeros_like(v)
    mask_r = (max_val == r)
    mask_g = (max_val == g)
    mask_b = (max_val == b)

    h[mask_r] = (g[mask_r] - b[mask_r]) / diff[mask_r]
    h[mask_g] = 2.0 + (b[mask_g] - r[mask_g]) / diff[mask_g]
    h[mask_b] = 4.0 + (r[mask_b] - g[mask_b]) / diff[mask_b]

    h = (h / 6.0) % 1.0  # Normalize to [0, 1]
    return h, s, v

def hsv_to_rgb(h, s, v):
    """
    PyTorch 版 HSV -> RGB 变换
    Input: H, S, V tensors [B, H, W], Range [0, 1]
    Output: [B, 3, H, W]
    """
    h_6 = h * 6.0
    i = torch.floor(h_6)
    f = h_6 - i
    
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    
    i = i % 6
    
    rgb = torch.zeros(h.shape[0], 3, h.shape[1], h.shape[2], device=h.device)
    
    mask = (i == 0); rgb[:, 0][mask], rgb[:, 1][mask], rgb[:, 2][mask] = v[mask], t[mask], p[mask]
    mask = (i == 1); rgb[:, 0][mask], rgb[:, 1][mask], rgb[:, 2][mask] = q[mask], v[mask], p[mask]
    mask = (i == 2); rgb[:, 0][mask], rgb[:, 1][mask], rgb[:, 2][mask] = p[mask], v[mask], t[mask]
    mask = (i == 3); rgb[:, 0][mask], rgb[:, 1][mask], rgb[:, 2][mask] = p[mask], q[mask], v[mask]
    mask = (i == 4); rgb[:, 0][mask], rgb[:, 1][mask], rgb[:, 2][mask] = t[mask], p[mask], v[mask]
    mask = (i == 5); rgb[:, 0][mask], rgb[:, 1][mask], rgb[:, 2][mask] = v[mask], p[mask], q[mask]
    
    return rgb

def gaussian_kernel(size, sigma, device):
    """生成一维高斯核矩阵"""
    coords = torch.arange(size, dtype=torch.float32, device=device)
    coords -= size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g /= g.sum()
    # 必须返回 [1, 1, size] 以配合上面的 apply_gaussian_blur
    return g.unsqueeze(0).unsqueeze(0)

def apply_gaussian_blur(tensor, kernel_size=21, sigma=3.0):
    """
    对 Tensor 进行高斯模糊 (Conv2d 实现) - 修复维度广播错误版
    """
    b, c, h, w = tensor.shape
    
    # 1. 获取基础高斯核 k, 原始形状: [1, 1, kernel_size]
    # 例如: [1, 1, 21]
    k = gaussian_kernel(kernel_size, sigma, tensor.device)
    
    # 2. 构建 X 方向卷积核 (水平)
    # 我们需要形状: [C, 1, 1, kernel_size]
    # 先 view 成 [1, 1, 1, 21], 再 expand
    kernel_x = k.view(1, 1, 1, kernel_size).expand(c, 1, 1, kernel_size)
    
    # 3. 构建 Y 方向卷积核 (垂直) -> 【关键修复点】
    # 我们需要形状: [C, 1, kernel_size, 1]
    # 先 view 成 [1, 1, 21, 1], 再 expand
    # 这样 21 就到了高度维度，宽度维度变成了 1，符合逻辑
    kernel_y = k.view(1, 1, kernel_size, 1).expand(c, 1, kernel_size, 1)
    
    # 4. 镜像填充 (Reflect Padding)
    pad = kernel_size // 2
    padded = F.pad(tensor, (pad, pad, pad, pad), mode='reflect')
    
    # 5. 可分离卷积 (Separable Convolution)
    # 第一步：水平卷积 (X)
    # 输入: [B, C, H+2pad, W+2pad] -> 输出: [B, C, H+2pad, W]
    blurred_x = F.conv2d(padded, kernel_x, groups=c)
    
    # 第二步：垂直卷积 (Y)
    # 输入: [B, C, H+2pad, W] -> 输出: [B, C, H, W]
    blurred_y = F.conv2d(blurred_x, kernel_y, groups=c)
    
    return blurred_y

def generate_cheek_mask(face_mask, width, height, device):
    """
    矩阵生成双颊高斯分布 (完全复刻 make_it_human_again 逻辑)
    """
    # 1. 计算质心 (Moment)
    grid_y, grid_x = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing='ij')
    
    # 仅在 face_mask > 0.5 的区域计算
    active_mask = (face_mask.squeeze() > 0.5).float()
    mass = active_mask.sum()
    
    if mass == 0: return torch.zeros_like(face_mask)
    
    cy = (grid_y * active_mask).sum() / mass
    cx = (grid_x * active_mask).sum() / mass
    
    # 2. 定义偏移量 (参考你的 OpenCV 代码)
    off_x = width * 0.16
    off_y = height * 0.03
    sigma = width * 0.08
    
    # 3. 计算左右脸颊的高斯分布矩阵
    # Left Cheek
    dist_l = (grid_x - (cx - off_x))**2 + (grid_y - (cy + off_y))**2
    l_cheek = torch.exp(-dist_l / (2 * sigma**2))
    
    # Right Cheek
    dist_r = (grid_x - (cx + off_x))**2 + (grid_y - (cy + off_y))**2
    r_cheek = torch.exp(-dist_r / (2 * sigma**2))
    
    # 4. 融合并受 Face Mask 约束
    cheek_mask = torch.maximum(l_cheek, r_cheek).unsqueeze(0).unsqueeze(0)
    return cheek_mask * face_mask

def get_segmentation_mask(image_pil, net, device, target_part='hair'):
    # 预处理
    to_tensor_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    # Resize 到 512x512 进行推理 (FaceParsing 标准)
    w, h = image_pil.size
    img_resized = image_pil.resize((512, 512), Image.BILINEAR)
    img_tensor = to_tensor_transform(img_resized).unsqueeze(0).to(device)
    
    with torch.no_grad():
        out = net(img_tensor)[0]
        # parsing shape: [512, 512]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)
        
    # 定义部位索引 (Map)
    # 17: hair, 1: skin, 10: nose (face通常结合1和10)
    indices = []
    if target_part == 'hair':
        indices = [17]
    elif target_part == 'face':
        indices = [1, 10] 
    elif target_part == 'clothes':
        indices = [16]
    
    # 生成mask
    mask = np.zeros_like(parsing, dtype=np.float32)
    for idx in indices:
        mask[parsing == idx] = 1.0
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    
    mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(device)
    return mask_tensor

def apply_matrix_color_edit(image_tensor, mask_tensor, target_color_hex, intensity=1.0, part_type='hair'):
    """
    功能：基于矩阵运算的特征编辑 (头发漂白/面部腮红)
    返回：Result Tensor, Debug Dictionary (for Heatmaps)
    """
    device = image_tensor.device
    B, C, H, W = image_tensor.shape
    
    # 1. 颜色空间转换 (RGB -> HSV)
    h_src, s_src, v_src = rgb_to_hsv(image_tensor) # All [B, H, W]
    
    # 解析目标颜色 HEX -> HSV
    rgb_tuple = tuple(int(target_color_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    target_rgb_t = torch.tensor(rgb_tuple, device=device).view(1, 3, 1, 1) / 255.0
    h_tgt, s_tgt, v_tgt = rgb_to_hsv(target_rgb_t)
    h_tgt_val = h_tgt.mean().item()

    # 初始化 Debug 信息
    debug_info = {
        'Original V': v_src.clone(),
        'Original S': s_src.clone(),
    }

    # ==========================================
    # 分支 A: 头发处理 (Digital Bleaching Logic)
    # ==========================================
    if part_type == 'hair':
        # 1. Mask 预处理: 腐蚀 (Erosion) -> 模糊 (Blur)
        # 腐蚀：使用 MaxPool 的负值模拟 MinPool
        mask_eroded = -F.max_pool2d(-mask_tensor, kernel_size=3, stride=1, padding=1)
        mask_soft = apply_gaussian_blur(mask_eroded, kernel_size=21, sigma=3.0)
        
        # 2. 矩阵运算: HSV 修改
        # Hue: 强行锁定
        h_new = torch.where(mask_soft.squeeze(1) > 0.05, torch.tensor(h_tgt_val, device=device), h_src)
        
        # Saturation: 暴力拉高 (对应 clip(S + 150))
        # PyTorch 0-1 空间：150/255 ≈ 0.6
        s_boost = s_src + 0.6 
        s_new = torch.where(mask_soft.squeeze(1) > 0.05, torch.clamp(s_boost, 0.0, 1.0), s_src)
        
        # Value: 压缩对比度并提亮 (对应 V * 0.5 + 120)
        # PyTorch 0-1 空间：120/255 ≈ 0.47
        v_bright = v_src * 0.5 + 0.47
        v_new = torch.where(mask_soft.squeeze(1) > 0.05, torch.clamp(v_bright, 0.0, 1.0), v_src)
        
        debug_info['Processed Mask'] = mask_soft
        debug_info['Target V'] = v_bright
        
        # 融合因子
        blend_mask = mask_soft

    # ==========================================
    # 分支 B: 面部处理 (Make It Human Logic)
    # ==========================================
    elif part_type == 'face':
        # 1. 动态生成双颊高斯 Mask
        cheek_mask = generate_cheek_mask(mask_tensor, W, H, device)
        mask_soft = cheek_mask # 这里的 mask 带有高斯渐变权重
        
        # 2. 矩阵运算: HSV 修改 (复刻 make_it_human_again)
        # Hue: 偏移向红色 (OpenCV H-8 约等于 PyTorch Hue -0.05)
        # 这里我们让它稍微偏红
        h_new = h_src # 保持肤色基调，或者微调
        
        # Saturation: 显著提高 (S + 100 * mask)
        # PyTorch: 100/255 ≈ 0.4
        s_boost = s_src + (0.4 * mask_soft.squeeze(1) * intensity)
        s_new = torch.clamp(s_boost, 0.0, 1.0)
        
        # Value: 稍微降低以增加血色深度 (V - 10 * mask)
        # PyTorch: 10/255 ≈ 0.04
        v_dim = v_src - (0.04 * mask_soft.squeeze(1) * intensity)
        v_new = torch.clamp(v_dim, 0.0, 1.0)
        
        debug_info['Processed Mask'] = mask_soft # 这是高斯腮红图
        debug_info['Target S'] = s_new

        blend_mask = mask_soft

    else:
        # 默认衣服等
        mask_soft = mask_tensor
        h_new = torch.where(mask_tensor.squeeze(1)>0.5, torch.tensor(h_tgt_val, device=device), h_src)
        s_new = s_src
        v_new = v_src * (1 - intensity * 0.2) # 简单变暗
        blend_mask = mask_tensor

    # 3. 合成与重建
    rgb_new = hsv_to_rgb(h_new, s_new, v_new)
    
    # 4. 最终软融合
    output = image_tensor * (1 - blend_mask) + rgb_new * blend_mask
    
    # 记录 Debug 用于热力图
    debug_info['Final H'] = h_new
    debug_info['Final S'] = s_new
    debug_info['Final V'] = v_new
    
    return torch.clamp(output, 0, 1), debug_info
