import os
os.environ["PYOPENGL_PLATFORM"] = "egl" # 设置 PyOpenGL 后端为 EGL，常用于无头（headless）渲染
import argparse # 用于命令行参数解析
import json
import random
import pickle # 用于序列化和反序列化 Python 对象
import pyrender # 用于高性能 3D 渲染
import trimesh # 用于处理和操作 3D 网格
import torch # PyTorch 库
import numpy as np
from PIL import Image # Pillow 库，用于图像处理
from typing import List
from natsort import natsorted # 自然排序
from pyquaternion import Quaternion as Q # 四元数库

from utils.visualize import frame2mp4 # 将图像序列转换为 MP4 视频的实用函数
from utils.misc import smplx_neutral_model, get_meshes_from_smplx # SMPL-X 模型和网格获取函数
from utils.visualize import skeleton_to_mesh # 将骨架关节转换为网格的实用函数

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
smplx_neutral_model = smplx_neutral_model.to(device=device) # 将 SMPL-X 中性模型加载到指定设备

from smplkit.constants import SKELETON_CHAIN
kinematic_chain = SKELETON_CHAIN.SMPLH['kinematic_chain'] # 获取 SMPL-H 模型的运动链 (kinematic chain)

def render_meshes_to_animation(save_path: str, meshes: List, appendix_meshes: List=None):
    """
    将网格渲染为动画视频。

    Args:
        save_path: 视频文件的保存路径（.mp4）。
        meshes: 要渲染的身体网格列表（每一帧一个）。
        appendix_meshes: 要渲染的附加网格列表（如场景或坐标轴）。
    """
    save_img_dir = os.path.join(os.path.dirname(save_path), 'img')
    os.makedirs(save_img_dir, exist_ok=True)

    ## camera setup (相机设置)
    H, W = 1080, 1080 # 渲染图像的高度和宽度
    camera_center = np.array([540.0, 540.0])
    # 使用内参相机模型，设置焦距 fx/fy 和中心 cx/cy
    camera = pyrender.camera.IntrinsicsCamera(
        fx=1060, fy=1060,
        cx=camera_center[0], cy=camera_center[1])
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.6) # 设置平行光

    ## camera pose (相机姿态)
    camera_pose = np.eye(4)
    camera_pose[0:3, -1] = np.array([0.0, 0.0, 6.5]) # 设置相机位置 (Z 轴 6.5)
    # 保持默认方向（没有旋转）
    camera_pose = camera_pose @ Q(axis=[1, 0, 0], angle=0).transformation_matrix

    ## rendering (渲染过程)
    for i in range(len(meshes)): # 遍历每一帧
        scene = pyrender.Scene()
        scene.add(camera, pose=camera_pose) # 添加相机
        scene.add(light, pose=camera_pose) # 添加灯光

        # 添加人体网格
        body_mesh = pyrender.Mesh.from_trimesh(meshes[i], smooth=False)
        scene.add(body_mesh)
        # 添加附加网格（场景、坐标轴等）
        for m in appendix_meshes:
            am = pyrender.Mesh.from_trimesh(m, smooth=False)
            scene.add(am)

        # 离屏渲染器
        r = pyrender.OffscreenRenderer(
            viewport_width=W,
            viewport_height=H,
        )
        color, _ = r.render(scene) # 渲染颜色图像
        color = color.astype(np.float32) / 255.0
        img = Image.fromarray((color * 255).astype(np.uint8))
        r.delete() # 释放渲染器资源
        save_img_path = os.path.join(save_img_dir, f'{i:03d}.png')
        img.save(save_img_path) # 保存单帧图像

    # 将图像序列转换为 MP4 视频并清理临时图像文件夹
    frame2mp4(os.path.join(save_img_dir, '%03d.png'), save_path)
    os.system(f"rm -rf {save_img_dir}")

def rendering(file_path, save_path, render_joint=False):
    """
    加载数据文件，准备网格并调用渲染函数。

    Args:
        file_path: 包含动作数据的 pickle 文件路径。
        save_path: 视频的基础保存路径。
        render_joint: 是否渲染骨架关节（True）或 SMPL-X 网格（False）。
    """
    with open(file_path, 'rb') as fp:
        data = pickle.load(fp) # 从 pickle 文件加载数据

    # 从加载的数据中提取关键信息
    joints = data['joints'] # 关节位置
    params = data['params'] # SMPL-X 参数
    text = data['text'] # 文本描述
    index = data['index']
    scene_trans = data['scene_trans'] # 场景变换矩阵
    scene_mesh = data['scene_mesh'] # 场景网格文件路径

    ## save path (处理保存路径)
    base_name = os.path.basename(save_path)
    assert int(base_name) == index # 确保文件名中的索引与数据中的索引一致
    # 构造最终的视频保存路径，包含文本描述
    save_path = (save_path + f'-{text[0:112]}.mp4').replace(' ', '_')

    ## smplx body meshes (处理 SMPL-X 身体网格)
    if render_joint:
        # 渲染骨架（22 个关节）
        body_meshes = skeleton_to_mesh(joints.reshape(-1, 22, 3), kinematic_chain)
    else:
        # 渲染 SMPL-X 网格
        params_tensor = torch.from_numpy(params).to(device=device).unsqueeze(0)
        # 从 SMPL-X 参数获取顶点和面
        verts, faces = get_meshes_from_smplx(smplx_neutral_model, params_tensor)
        verts = verts.squeeze(0).cpu().numpy()

        # 创建 trimesh 对象列表
        body_meshes = [trimesh.Trimesh(vertices=verts[i], faces=faces) for i in range(len(verts))]

    ## scene mesh (处理场景网格)
    print(f"--- 正在加载场景: {scene_mesh} ---")
    # 加载场景网格（通常是 .ply 或 .obj 文件）
    scene_mesh = trimesh.load(scene_mesh, process=False)
    # 应用场景变换矩阵（将场景放置到正确的位置和方向）
    scene_mesh.apply_transform(scene_trans)

    # 调用渲染函数
    render_meshes_to_animation(
        save_path,
        body_meshes,
        # 附加网格包括场景网格和表示世界坐标系的小坐标轴
        appendix_meshes=[scene_mesh, trimesh.creation.axis(0.05)],
    )

    return body_meshes, scene_mesh

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 定义命令行参数
    parser.add_argument('--folder', type=str, default='') # 动作数据文件夹
    parser.add_argument('--file', type=str, default='') # 单个动作数据文件
    parser.add_argument('--cnt', type=int, default=30) # 处理文件数量上限
    parser.add_argument('--save_mesh', action='store_true') # 是否保存人体网格
    parser.add_argument('--save_scene', action='store_true') # 是否保存场景网格
    parser.add_argument('--render_joint', action='store_true') # 是否渲染骨架
    args = parser.parse_args()

    files = []
    # 根据命令行参数构建待处理的文件列表
    if args.folder != '' and os.path.exists(args.folder):
        files = natsorted(os.listdir(args.folder))
        random.seed(0)
        random.shuffle(files) # 随机打乱文件列表
        files = [os.path.join(args.folder, file) for file in files]
    elif args.file != '' and os.path.exists(args.file):
        files = [args.file]
    else:
        raise ValueError('Invalid path or folder')

    # 遍历文件列表并执行渲染
    for f in files[0:args.cnt]:
        prefix = os.path.dirname(os.path.dirname(f))
        basename = os.path.basename(f).split('.')[0]

        # 渲染视频
        body_meshes, scene_mesh = rendering(f, os.path.join(prefix, 'video', f'{basename}'), args.render_joint)

        # 可选：保存网格文件
        if args.save_mesh:
            save_meshes(body_meshes, os.path.join(prefix, 'meshes'), basename, args.render_joint)
        if args.save_scene:
            scene_mesh.export(os.path.join(prefix, 'meshes', basename, 'scene.ply'))