import sys
import os
import importlib
import inspect

# ------------------------------
# 1️⃣ 自动定位 X-StereoLab 根目录
# ------------------------------
# 假设当前脚本在 PythonProject 根目录
PROJECT_DIR = os.getcwd()  # 当前工作目录
STEREOLAB_DIR = os.path.join(PROJECT_DIR, "X-StereoLab")

if not os.path.exists(STEREOLAB_DIR):
    raise FileNotFoundError(f"X-StereoLab 目录不存在: {STEREOLAB_DIR}")

# ------------------------------
# 2️⃣ 将 X-StereoLab 根目录加入 sys.path
# ------------------------------
sys.path.append(STEREOLAB_DIR)

# ------------------------------
# 3️⃣ 检查 dsgn 是否存在
# ------------------------------
DSGN_DIR = os.path.join(STEREOLAB_DIR, "dsgn")
if os.path.exists(DSGN_DIR):
    sys.path.append(DSGN_DIR)
    print("dsgn 目录加入路径成功")
else:
    print("警告：dsgn 目录不存在，部分模块可能无法导入")

# ------------------------------
# 4️⃣ 要导入的模块列表
# ------------------------------
modules_to_import = [
    "disparity.models.stereonet",
    "disparity.models.stereonet_disp"
]

# ------------------------------
# 5️⃣ 动态导入模块并列出类
# ------------------------------
for module_name in modules_to_import:
    try:
        m = importlib.import_module(module_name)
        classes = [name for name, obj in inspect.getmembers(m) if inspect.isclass(obj)]
        print(f"Module: {module_name} => classes: {classes}")
    except Exception as e:
        print(f"Module: {module_name} import failed: {str(e)[:200]}")

# ------------------------------
# 6️⃣ 测试导入 dsgn
extra_compile_args={'cxx': ['-std=c++14']}

# ------------------------------
try:
    import dsgn
    print("dsgn 模块导入成功！")
except Exception as e:
    print(f"dsgn 模块导入失败: {str(e)[:200]}")
