"""Real local inference. Scores are uncalibrated model responses, not authenticity proof."""
import hashlib
import importlib.util
import json
import math
import tempfile
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / 'models'
VERSION = 'local-v1'


def device():
    import torch
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


@lru_cache(maxsize=1)
def ai_model():
    import timm
    import torch
    from safetensors.torch import load_file
    torch.set_num_threads(4)
    path = MODELS / 'commfor-384.safetensors'
    if not path.is_file():
        raise FileNotFoundError('缺少Community Forensics权重')
    model = timm.create_model('vit_small_patch16_384.augreg_in21k_ft_in1k', pretrained=False, num_classes=1)
    state = load_file(str(path))
    # Official wrapper names the backbone `vit`.
    state = {k.removeprefix('vit.'): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return model.eval().to(device())


def ai_detect(path):
    import torch
    from torchvision import transforms
    transform = transforms.Compose([transforms.Resize(440), transforms.CenterCrop(384), transforms.ToTensor(),
                                    transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    im = ImageOps.exif_transpose(Image.open(path)).convert('RGB')
    with torch.inference_mode():
        score = torch.sigmoid(ai_model()(transform(im).unsqueeze(0).to(device()))).item()
    if score >= .8:
        verdict = '整图 AI 生成迹象较强（模型提示）'
    elif score <= .2:
        verdict = '整图 AI 生成迹象较弱；不排除局部修改'
    else:
        verdict = '整图 AI 检测不确定'
    return {'status':'已运行', 'model':'Community Forensics 384', 'score':score, 'verdict':verdict,
            'device':device(), 'preprocess':'官方Resize(440)+CenterCrop(384)，可能未覆盖位于画面边缘的仪表',
            'note':'分数未经测温业务校准；阈值0.2/0.8仅作演示，不是AI概率。'}


@lru_cache(maxsize=1)
def cat_model():
    import torch
    import yaml
    from yacs.config import CfgNode
    torch.set_num_threads(4)
    folder = ROOT / 'vendor' / 'catnet'
    spec = importlib.util.spec_from_file_location('catnet_network', folder / 'network_CAT.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = CfgNode(yaml.safe_load((folder / 'CAT_full.yaml').read_text()))
    model = mod.CAT_Net(cfg)
    # Legacy author checkpoint includes NumPy scalar training metadata. Allow only these concrete safe types.
    with torch.serialization.safe_globals([(np._core.multiarray.scalar, 'numpy.core.multiarray.scalar'), np.dtype, type(np.dtype('float64')), type(np.dtype('float32'))]):
        ckpt = torch.load(MODELS / 'CAT_full_v2.pth.tar', map_location='cpu', weights_only=True)
    model.load_state_dict(ckpt['state_dict'], strict=True)
    return model.eval().to(device())


def jpeg_features(path):
    """Preserve original JPEG DCT grid; follow official conversion for non-JPEG input."""
    import jpeglib
    im = Image.open(path)
    converted = im.format != 'JPEG'
    temp = None
    if converted:
        temp = tempfile.TemporaryDirectory()
        source = Path(temp.name) / 'converted.jpg'
        im.convert('RGB').save(source, quality=100, subsampling=0)
    else:
        source = path
    jpg = jpeglib.read_dct(str(source))
    rgb = np.array(Image.open(source).convert('RGB'), dtype=np.float32)
    y = jpg.Y
    dct = y.transpose(0,2,1,3).reshape(y.shape[0]*8,y.shape[1]*8).copy()
    q = jpg.qt[jpg.quant_tbl_no[0]].astype(np.float32).copy()
    if temp:
        temp.cleanup()
    im.close()
    return rgb, dct, q, converted


def tile_starts(length, tile=512, stride=384):
    if length <= tile:
        return [0]
    last = math.ceil((length-tile)/8)*8
    return sorted(set(list(range(0,last,stride)) + [last]))


def local_detect(path, out, progress=None):
    import torch
    import torch.nn.functional as F
    start = time.monotonic()
    model = cat_model()
    rgb, dct, q, converted = jpeg_features(path)
    h,w = rgb.shape[:2]
    ys,xs = tile_starts(h),tile_starts(w)
    total = len(ys)*len(xs)
    if total > 100:
        raise ValueError('图片过大：需要超过100块推理，请人工处理；未生成检测结果')
    accum = np.zeros((h,w),np.float32)
    counts = np.zeros((h,w),np.float32)
    tile=512
    qt = torch.from_numpy(q).reshape(1,1,8,8).to(device())
    n=0
    for y in ys:
        for x in xs:
            rh,rw = min(tile,h-y),min(tile,w-x)
            patch=np.full((tile,tile,3),127.5,np.float32)
            patch[:rh,:rw]=rgb[y:y+rh,x:x+rw]
            coef=np.zeros((tile,tile),np.int32)
            dh,dw=min(tile,dct.shape[0]-y),min(tile,dct.shape[1]-x)
            coef[:dh,:dw]=dct[y:y+dh,x:x+dw]
            values=np.minimum(np.abs(coef),20)
            volume=(values[None,:,:] == np.arange(21)[:,None,None]).astype(np.float32)
            tensor=np.concatenate([(patch.transpose(2,0,1)-127.5)/127.5,volume])
            with torch.inference_mode():
                pred=F.softmax(model(torch.from_numpy(tensor).unsqueeze(0).to(device()),qt),dim=1)[:,1:2]
                pred=F.interpolate(pred,size=(tile,tile),mode='bilinear',align_corners=False)[0,0].cpu().numpy()
            accum[y:y+rh,x:x+rw]+=pred[:rh,:rw]
            counts[y:y+rh,x:x+rw]+=1
            n+=1
            if progress:progress(n,total)
    scores=accum/np.maximum(counts,1)
    # Transform outputs to match EXIF-transposed display, not the JPEG coefficient grid.
    with Image.open(path) as source_image:
        orientation=source_image.getexif().get(274,1)
    transforms={2:Image.Transpose.FLIP_LEFT_RIGHT,3:Image.Transpose.ROTATE_180,4:Image.Transpose.FLIP_TOP_BOTTOM,
                5:Image.Transpose.TRANSPOSE,6:Image.Transpose.ROTATE_270,7:Image.Transpose.TRANSVERSE,8:Image.Transpose.ROTATE_90}
    if orientation in transforms:
        scores=np.asarray(Image.fromarray(scores).transpose(transforms[orientation])).copy()
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    np.save(out/'scores.npy',scores)
    # Fixed [0,1] palette; never min-max normalize to exaggerate weak responses.
    heat=np.stack([scores, .15+ .55*(1-np.abs(2*scores-1)), 1-scores],axis=-1)
    heat=Image.fromarray(np.clip(heat*255,0,255).astype('uint8'))
    original=ImageOps.exif_transpose(Image.open(path)).convert('RGB')
    overlay=Image.blend(original,heat,.45)
    overlay.save(out/'heatmap.png')
    high=float((scores>=.5).mean())
    result={'status':'已运行','model':'CAT-Net v2','heatmap':str((out/'heatmap.png').resolve()),
            'scores_path':str((out/'scores.npy').resolve()),'high_response_fraction':high,
            'p95_response':float(np.quantile(scores,.95)),'device':device(),'elapsed_seconds':round(time.monotonic()-start,2),
            'tiles':total,'converted_from_non_jpeg':converted,
            'verdict':'存在局部编辑响应，需查看是否涉及读数' if high>=.01 else '未发现大面积强编辑响应；不能排除小面积改数字',
            'message':'红色表示编辑响应较强，蓝色较弱。红框、水印也可能触发；不能直接证明温度数字被改。',
            'preprocess':'原分辨率512像素切块，384步长，保持8×8 JPEG网格，重叠平均；非官方整图评测协议。'+('非JPEG按官方做法转JPEG，压缩取证可靠性受限。' if converted else '')}
    (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    return result


def region_evidence(result, box):
    scores=np.load(result['scores_path'],allow_pickle=False)
    h,w=scores.shape
    l,t,r,b=box
    region=scores[max(0,int(t*h)):min(h,int(b*h)),max(0,int(l*w)):min(w,int(r*w))]
    if not region.size:raise ValueError('屏幕区域为空')
    return {'mean_response':float(region.mean()),'high_response_fraction':float((region>=.5).mean()),
            'verdict':'选定屏幕区域存在编辑响应，需人工核查' if (region>=.5).mean()>=.05 else '选定屏幕未出现足够强的编辑响应，不能据此证明未修改'}


def readiness():
    ai = MODELS/'commfor-384.safetensors'
    return {'ai':ai.is_file() and ai.stat().st_size == 87262324,'local':(MODELS/'CAT_full_v2.pth.tar').is_file()}
