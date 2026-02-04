import io  # 标准库：处理字节流/内存流
import os  # 标准库：访问文件路径和环境变量
from typing import Tuple  # 类型注解：用于标注元组返回类型

import numpy as np  # 第三方库：数组处理
import torch  # PyTorch：深度学习框架
import torch.nn.functional as F  # PyTorch：常用函数（如插值）
from fastapi import FastAPI, File, Form, UploadFile  # FastAPI：Web API 相关类型
from fastapi.responses import HTMLResponse, Response  # FastAPI：响应类型
from fastapi.staticfiles import StaticFiles  # FastAPI：静态文件挂载
from PIL import Image  # Pillow：图像处理
from torchvision import transforms  # torchvision：图像预处理
from transformers import AutoModelForImageSegmentation, PreTrainedModel  # Transformers：模型加载

APP_ROOT = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在目录
STATIC_DIR = os.path.join(APP_ROOT, "static")  # 静态资源目录

app = FastAPI()  # 创建 FastAPI 应用实例
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")  # 挂载静态资源路径

MODEL_CACHE = {}  # 模型缓存，避免重复加载


def get_device() -> torch.device:  # 根据环境选择推理设备
    if torch.cuda.is_available():  # 如果有 CUDA GPU
        return torch.device("cuda")  # 使用 CUDA
    if torch.backends.mps.is_available():  # 如果是 Apple 芯片且支持 MPS
        return torch.device("mps")  # 使用 MPS
    return torch.device("cpu")  # 否则使用 CPU


def resolve_model_source(model_name: str) -> Tuple[str, str, bool]:  # 解析模型来源
    """
    Resolve model source.
    Returns (repo_id, local_dir, local_only).
    """
    model_name = (model_name or "").strip() or "portrait"  # 规范化模型名，默认 portrait

    if "/" in model_name:  # 如果包含斜杠，视为 HF 仓库 ID
        return model_name, "", os.getenv("LOCAL_ONLY") == "1"  # 返回仓库ID

    base_dir = os.getenv("MODEL_BASE_DIR", "")  # 读取本地模型根目录
    if base_dir:  # 如果配置了本地模型目录
        local_dir = os.path.join(base_dir, model_name)  # 拼出本地模型目录
        return "", local_dir, True  # 本地加载，强制本地

    repo_id = os.getenv("MODEL_REPO", "ZhengPeng7/BiRefNet-portrait")  # 默认仓库
    return repo_id, "", os.getenv("LOCAL_ONLY") == "1"  # 返回远端仓库或本地开关


def load_model(model_name: str) -> PreTrainedModel:  # 加载并缓存模型
    if model_name in MODEL_CACHE:  # 如果已加载过
        return MODEL_CACHE[model_name]  # 直接复用缓存

    repo_id, local_dir, local_only = resolve_model_source(model_name)  # 解析模型来源
    if local_dir:  # 如果指定了本地目录
        if not os.path.isdir(local_dir):  # 本地目录不存在
            raise FileNotFoundError(  # 抛出错误
                f"MODEL_BASE_DIR is set, but model folder not found: {local_dir}"
            )
        model = AutoModelForImageSegmentation.from_pretrained(  # 从本地加载模型
            local_dir,  # 本地目录
            local_files_only=True,  # 仅使用本地文件
            trust_remote_code=True,  # 允许加载自定义代码
        )
    else:  # 否则从远端加载
        model = AutoModelForImageSegmentation.from_pretrained(  # 从 Hugging Face 加载模型
            repo_id,  # 远端仓库 ID
            trust_remote_code=True,  # 允许加载自定义代码
            local_files_only=local_only,  # 根据开关决定是否仅本地
        )

    device = get_device()  # 获取推理设备
    model.to(device)  # 把模型移动到设备
    model.eval()  # 切换到推理模式
    MODEL_CACHE[model_name] = model  # 缓存模型
    return model  # 返回模型


def preprocess(image: Image.Image, size: Tuple[int, int]) -> torch.Tensor:  # 图像预处理
    transform = transforms.Compose(  # 创建预处理管线
        [
            transforms.Resize(size),  # 调整到固定尺寸
            transforms.ToTensor(),  # 转为张量
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # 标准化
        ]
    )
    return transform(image)  # 应用预处理


def birefnet_predict(image: Image.Image, model_name: str, putalpha: bool) -> Image.Image:  # 抠图主函数
    device = get_device()  # 获取设备
    model = load_model(model_name)  # 加载模型

    image = image.convert("RGB")  # 确保是 RGB
    image_tensor = preprocess(image, (1024, 1024)).unsqueeze(0).to(device)  # 预处理+加批次

    with torch.no_grad():  # 推理时不需要梯度
        preds = model(image_tensor)[-1].sigmoid().cpu()  # 得到预测并映射到 0~1

    w, h = image.size  # 原图宽高
    result = torch.squeeze(F.interpolate(preds, size=(h, w)))  # 将掩膜放回原尺寸
    ma = torch.max(result)  # 最大值
    mi = torch.min(result)  # 最小值
    denom = (ma - mi) if (ma - mi) > 0 else torch.tensor(1.0)  # 防止除零
    result = (result - mi) / denom  # 归一化到 0~1
    mask_array = (result * 255).numpy().astype(np.uint8)  # 转为 0~255 的灰度数组
    mask = Image.fromarray(mask_array)  # 转为 Pillow 图像

    if putalpha:  # 如果要输出带透明通道
        out = image.copy()  # 复制原图
        out.putalpha(mask)  # 用 mask 作为 alpha
        return out  # 返回透明背景图

    empty = Image.new("RGBA", image.size, 0)  # 创建透明底图
    return Image.composite(image, empty, mask)  # 合成前景抠图结果


@app.get("/", response_class=HTMLResponse)  # 首页路由
def index() -> HTMLResponse:  # 返回静态 HTML
    with open(os.path.join(STATIC_DIR, "index.html"), "r", encoding="utf-8") as f:  # 读取 HTML
        return HTMLResponse(f.read())  # 返回 HTML 内容


@app.post("/api/rmbg")  # 抠图 API
def rmbg(  # 处理抠图请求
    image: UploadFile = File(...),  # 上传图片文件
    model: str = Form("portrait"),  # 模型名称表单参数
    putalpha: str = Form("true"),  # 是否输出透明通道
) -> Response:  # 返回响应
    raw = image.file.read()  # 读取上传文件字节
    img = Image.open(io.BytesIO(raw))  # 从字节流打开图像
    putalpha_bool = putalpha.lower() not in {"0", "false", "no"}  # 转为布尔值
    out = birefnet_predict(img, model, putalpha_bool)  # 执行抠图
    buf = io.BytesIO()  # 创建内存缓冲区
    out.save(buf, format="PNG")  # 保存为 PNG 到缓冲区
    return Response(content=buf.getvalue(), media_type="image/png")  # 返回 PNG 响应


if __name__ == "__main__":  # 直接运行脚本时执行
    import uvicorn  # 仅在直跑时导入服务器

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)  # 启动开发服务器
