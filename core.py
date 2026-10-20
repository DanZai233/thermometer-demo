import base64
import hashlib
import io
import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import imagehash
import requests
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
PROMPT = '''你是测温照片的信息提取助手，不判断图片是否篡改。只返回JSON对象，不使用markdown。字段：temperature（数字或null），unit（C/F/unknown），instrument（仪表类型），measurement（air/surface/plant/unknown；分别为空气温度、表面温度、热力站参数、不确定），watermark_date（YYYY-MM-DD或null），watermark_time（HH:MM或null），location（可见地点或null），readability（clear/uncertain/unreadable），notes（字符串数组）。只读取仪表显示，不用叠加文字代替读数；模糊不猜数字。温度计照片不能证明探针位置、测量稳定或拍摄真实性。'''


def ingest(raw, name):
    im = Image.open(io.BytesIO(raw))
    im.verify()
    im = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert('RGB')
    digest = hashlib.sha256(raw).hexdigest()
    DATA.mkdir(exist_ok=True)
    suffix = Image.open(io.BytesIO(raw)).format.lower()
    path = DATA / (digest + '.' + suffix)
    if not path.exists():
        path.write_bytes(raw)
    return {'name': Path(name).name, 'sha256': digest, 'path': str(path),
            'phash': str(imagehash.phash(im)), 'size': list(im.size),
            'facts': {}, 'source': '未提取', 'forensic': {'status': '未运行'}}


def preview(path):
    im = ImageOps.exif_transpose(Image.open(path)).convert('RGB')
    im.thumbnail((1600, 1600))
    b = io.BytesIO()
    im.save(b, format='JPEG', quality=90)
    return b.getvalue()


def extract(item, endpoint, model, key):
    # Endpoint and credentials are explicitly configured by the operator.
    url = endpoint.rstrip('/') + '/chat/completions'
    media = base64.b64encode(preview(item['path'])).decode()
    payload = {
        'model': model, 'messages': [{'role': 'user', 'content': [
            {'type': 'text', 'text': PROMPT},
            {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + media}}]}],
        'temperature': 0}
    if urlparse(endpoint).hostname == 'api.deepseek.com':
        if model != 'deepseek-flash':
            raise ValueError('本应用的DeepSeek官方视觉接口仅配置deepseek-flash')
        payload.update(thinking={'type': 'disabled'}, response_format={'type': 'json_object'}, max_tokens=2048)
        payload['messages'][0]['content'][1]['image_url']['detail'] = 'high'
    r = requests.post(url, headers={'Authorization': 'Bearer ' + key}, json=payload, timeout=90)
    r.raise_for_status()
    content = r.json()['choices'][0]['message']['content']
    content = re.sub(r'^```(?:json)?\s*|\s*```$', '', content.strip())
    facts = json.loads(content)
    if not isinstance(facts, dict):
        raise ValueError('模型未返回JSON对象')
    temp = facts.get('temperature')
    if temp is not None and (isinstance(temp, bool) or not isinstance(temp, (int, float))):
        raise ValueError('temperature必须是数字或null')
    if facts.get('measurement') not in ['air', 'surface', 'plant', 'unknown']:
        facts['measurement'] = 'unknown'
    item['facts'] = facts
    item['source'] = model + ' · API提取（待人工确认）'


def forensic(item, progress=None):
    # Optional local adapter contract: executable + input path + output directory.
    # No shell interpolation; failures never become successful detection results.
    executable = os.getenv('FORENSIC_ADAPTER')
    if not executable:
        from local_models import VERSION, local_detect, readiness
        if not readiness()['local']:
            return {'status': '未运行', 'message': 'CAT-Net权重尚未就绪；没有篡改判断。'}
        out = DATA / ('catnet_' + VERSION + '_' + item['sha256'])
        cached = out / 'result.json'
        if cached.is_file():
            result = json.loads(cached.read_text())
            if Path(result['heatmap']).is_file() and Path(result['scores_path']).is_file():
                return result
        return local_detect(item['path'], out, progress)
    out = DATA / ('forensic_' + item['sha256'])
    out.mkdir(exist_ok=True)
    p = subprocess.run([executable, item['path'], str(out)], capture_output=True, text=True, timeout=180)
    if p.returncode:
        raise RuntimeError('本地取证适配器执行失败')
    result = json.loads((out / 'result.json').read_text())
    heatmap = Path(result['heatmap']).resolve()
    if not heatmap.is_file() or out.resolve() not in heatmap.parents:
        raise ValueError('热力图必须是输出目录内的真实文件')
    Image.open(heatmap).verify()
    return {'status': '已运行', 'model': result.get('model', '本地模型'), 'heatmap': str(heatmap),
            'message': '异常响应仅供复核，不是篡改概率或真实性证明。'}


def checks(items, history):
    results = []
    for i, a in enumerate(items):
        f = a['facts']
        if f.get('measurement') == 'surface':
            results.append(f"{a['name']}：测量对象为表面，不能直接作为室温验收证据。")
        if f.get('measurement') == 'plant':
            results.append(f"{a['name']}：热力站参数不能替代住户室温。")
        if not f or f.get('temperature') is None or f.get('readability') != 'clear':
            results.append(f"{a['name']}：读数未确认或不清晰，建议人工核对或补拍。")
        for b in items[i + 1:]:
            distance = imagehash.hex_to_hash(a['phash']) - imagehash.hex_to_hash(b['phash'])
            if a['sha256'] == b['sha256']:
                results.append(f"{a['name']} 与 {b['name']}：同一文件重复上传。")
            elif distance <= 8:
                results.append(f"{a['name']} 与 {b['name']}：外观相似（pHash距离 {distance}），需核对是否复用；不是造假结论。")
            g = b['facts']
            for field, label in [('watermark_date', '水印日期'), ('location', '水印地点')]:
                if f.get(field) and g.get(field) and f[field] != g[field]:
                    results.append(f"{a['name']} 与 {b['name']}：{label}不同，核对是否属于同次处置。")
            if a['stage'] != b['stage'] and f.get('watermark_date') and f.get('watermark_time') and (f.get('watermark_date'), f.get('watermark_time')) == (g.get('watermark_date'), g.get('watermark_time')):
                results.append(f"{a['name']} 与 {b['name']}：前后阶段水印时间相同，不能确认维修先后。")
        for report in history:
            for old in report.get('images', []):
                if a['sha256'] == old['sha256']:
                    results.append(f"{a['name']}：与已保存工单 {report['case_id']} 文件相同；核对附件复用。")
                elif imagehash.hex_to_hash(a['phash']) - imagehash.hex_to_hash(old['phash']) <= 8:
                    results.append(f"{a['name']}：与已保存工单 {report['case_id']} 照片相似，需人工核对。")
    return list(dict.fromkeys(results))


def history():
    folder = DATA / 'reports'
    return [json.loads(p.read_text()) for p in folder.glob('*.json')] if folder.exists() else []


def save_report(report):
    folder = DATA / 'reports'
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / (datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.json')
    p.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return p


def ai_forensic(item):
    from local_models import VERSION, ai_detect, readiness
    if not readiness()['ai']:
        return {'status':'未运行','message':'AI检测权重尚未就绪'}
    folder = DATA / ('ai_' + VERSION + '_' + item['sha256'])
    folder.mkdir(exist_ok=True)
    cache = folder / 'result.json'
    if cache.is_file():
        return json.loads(cache.read_text())
    result = ai_detect(item['path'])
    cache.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    return result
