import json
import os
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
from PIL import Image, ImageOps

from core import ROOT, ai_forensic, checks, extract, forensic, history, ingest, save_report

st.set_page_config(page_title='测温证据审核', page_icon='🌡️', layout='wide')
st.markdown('''<style>h1{letter-spacing:-1px}
.block-container{max-width:1400px;padding-top:2rem}
div[data-testid="stMetric"]{background:transparent;border:1px solid #778498;border-radius:12px;padding:16px}
[data-testid="stCaptionContainer"]{color:inherit;opacity:0.9}
[data-testid="stWidgetLabel"]{color:inherit}
button:focus-visible,input:focus-visible,textarea:focus-visible{outline:3px solid #4c9aff!important;outline-offset:2px}</style>''', unsafe_allow_html=True)
st.caption('供热现场 · 证据审核工作台')
st.title('测温照片，逐项核对')
st.write('检测整图 AI 生成迹象与局部编辑，读取温度并核对维修证据。模型结果供人工复核。')
with st.sidebar:
    st.subheader('分析配置')
    provider = st.selectbox('模型供应商', ['DeepSeek 官方', '自定义兼容接口'])
    if provider == 'DeepSeek 官方':
        endpoint = 'https://api.deepseek.com'
        model = 'deepseek-flash'
        st.caption('deepseek-flash · 支持图片输入')
        st.caption('API：https://api.deepseek.com')
        key = st.text_input('DeepSeek API Key（仅本次会话使用）', value=os.getenv('DEEPSEEK_API_KEY', ''), type='password', key='deepseek_api_key')
    else:
        endpoint = st.text_input('OpenAI 兼容 API 地址', value=os.getenv('VISION_BASE_URL', 'https://api.deepseek.com'))
        model = st.text_input('视觉模型名称', value=os.getenv('VISION_MODEL', 'deepseek-flash'))
        key = st.text_input('API Key（仅本次会话使用）', value=os.getenv('VISION_API_KEY', ''), type='password', key='custom_api_key')
    external_ok = st.checkbox('允许将本次照片发送到配置的 API')
    st.caption('照片可能含人脸和地址。未勾选时不调用外部 API；上传、查重和人工审核均在本地。')
    st.divider()
    st.write('本地检测模型')
    from local_models import readiness
    available = readiness()
    st.caption('CAT-Net v2 · 权重文件就绪' if available['local'] else 'CAT-Net v2 · 权重未就绪')
    st.caption('Community Forensics · 权重文件就绪' if available['ai'] else '整图AI模型 · 权重未就绪')
    st.caption('本地检测不需要 API Key，也不上传照片。')
    st.caption('pHash 只筛选相似照片，不证明篡改；水印也不证明拍摄时间。')

case_id = st.text_input('工单编号', value='演示工单-001')
left, right = st.columns(2)
before = left.file_uploader('维修前照片', type=['jpg', 'jpeg', 'png', 'webp'], accept_multiple_files=True)
after = right.file_uploader('维修后照片', type=['jpg', 'jpeg', 'png', 'webp'], accept_multiple_files=True)

if 'items' not in st.session_state:
    st.session_state['items'] = []

def load_uploads():
    result = []
    for files, stage in [(before, '维修前'), (after, '维修后')]:
        for file in files:
            item = ingest(file.getvalue(), file.name)
            item['stage'] = stage
            result.append(item)
    return result

c1, c2 = st.columns([1, 3])
if c1.button('导入上传照片', type='primary', use_container_width=True):
    try:
        st.session_state['items'] = load_uploads()
        if not st.session_state['items']:
            st.warning('请先上传照片。')
    except Exception as e:
        st.error('导入失败：' + str(e))
if c2.button('加载提供的案例（人工标注演示）'):
    manifest = ROOT / 'samples' / 'manifest.json'
    if not manifest.exists():
        st.error('未找到演示样例。')
    else:
        result = []
        for row in json.loads(manifest.read_text()):
            item = ingest((manifest.parent / row['file']).read_bytes(), row['file'])
            item.update(stage='阶段待确认', facts=row['facts'], source='人工观察标注（不是模型检测）', screen_box_percent=row.get('screen_box_percent'))
            result.append(item)
        st.session_state['items'] = result

items = st.session_state['items']
if not items:
    st.info('先上传照片，或加载本地案例。案例阶段尚未确认，不自动推断维修前后。')
    st.stop()

warnings = checks(items, history())
st.warning('当前为未校准模型演示：在同图重编码与人为改数字对照中，CAT-Net未明显区分两者。可用于筛查线索，尚不能可靠判定温度数字是否被改。')
m1, m2, m3 = st.columns(3)
m1.metric('已导入照片', len(items))
m2.metric('需核对线索', len(warnings))
m3.metric('已运行局部取证', sum(x['forensic']['status'] == '已运行' for x in items))

def run_local(item):
    try:
        item['ai_detection'] = ai_forensic(item)
        if item['facts'].get('measurement') == 'plant':
            item['ai_detection']['domain_warning'] = '热力站显示屏属于特殊场景，整图AI检测可能误报；高响应不代表该照片由AI生成。'
        st.write(item['ai_detection'].get('verdict', item['ai_detection'].get('message', '未运行')))
    except Exception as e:
        item['ai_detection'] = {'status':'失败','message':str(e)}
        st.error('AI检测失败：' + str(e))
    bar = st.progress(0, text='准备局部检测')
    try:
        item['forensic'] = forensic(item, lambda n,total: bar.progress(n/total, text=f'CAT-Net分块推理 {n}/{total}'))
        st.write(item['forensic'].get('verdict', item['forensic'].get('message', '未运行')))
    except Exception as e:
        item['forensic'] = {'status':'失败','message':str(e)}
        st.error('局部检测失败：' + str(e))
    finally:
        bar.empty()

if st.button('检测 AI 生成 / 局部 P 图（本地运行）', type='primary'):
    for item in items:
        with st.status('本地检测 ' + item['name']):
            run_local(item)
    st.rerun()

if st.button('提取照片信息（DeepSeek / 自定义 API）'):
    for item in items:
        with st.status('分析 ' + item['name']):
            if key and external_ok:
                try:
                    extract(item, endpoint, model, key)
                    st.write('读数与水印提取完成，待人工确认。')
                except Exception as e:
                    st.error('API 提取失败：' + type(e).__name__ + '；保留已有标注。')
            else:
                st.write('未调用外部 API。可在照片下方人工录入。')

    st.rerun()

tab1, tab2, tab3 = st.tabs(['照片与读数', '一致性核对', '审核与导出'])
with tab1:
    for i in range(0, len(items), 2):
        cols = st.columns(2)
        for slot, (col, item) in enumerate(zip(cols, items[i:i+2])):
            with col:
                st.subheader(item['name'])
                st.image(item['path'], width='stretch')
                st.caption(item['source'])
                f = item['facts']
                st.write('温度：', str(f.get('temperature', '未确认')), ' · 单位：', f.get('unit', 'unknown'))
                st.write('水印日期：', f.get('watermark_date') or '未确认', ' · 时间：', f.get('watermark_time') or '未确认')
                st.write('地点：', f.get('location') or '未确认')
                ai = item.get('ai_detection', {'status':'未运行'})
                st.write('**整图 AI 检测：** ' + ai.get('verdict', ai['status']))
                if ai.get('score') is not None:
                    st.metric('AI 模型响应分数（未校准）', f"{ai['score']:.3f}")
                    st.caption(ai['note'])
                    st.caption(ai['preprocess'])
                if ai.get('domain_warning'):
                    st.warning(ai['domain_warning'])
                st.write('**局部 P 图检测：** ' + item['forensic'].get('verdict', item['forensic']['status']))
                st.info('全图局部编辑响应（含正常标注）：' + item['forensic']['status'] + '。' + item['forensic'].get('message', ''))
                if item['forensic'].get('heatmap'):
                    st.image(item['forensic']['heatmap'], caption='CAT-Net固定色标：红色响应高，蓝色响应低；水印和红框可能触发')
                    if 'high_response_fraction' in item['forensic']:
                        st.caption(f"强响应区域占比：{item['forensic']['high_response_fraction']:.1%} · 耗时 {item['forensic']['elapsed_seconds']}秒 · {item['forensic']['device']}")
                        st.caption(item['forensic']['preprocess'])
                with st.expander('屏幕放大 / 人工核对'):
                    im = ImageOps.exif_transpose(Image.open(item['path']))
                    default_box = item.get('screen_box_percent') or [20,40,80,80]
                    box = st.slider('屏幕区域：左、右边界（%）', 0, 100, (default_box[0], default_box[2]), key=f'boxx_{i+slot}_{item["sha256"]}')
                    vertical = st.slider('屏幕区域：上、下边界（%）', 0, 100, (default_box[1], default_box[3]), key=f'boxy_{i+slot}_{item["sha256"]}')
                    if box[0] < box[1] and vertical[0] < vertical[1]:
                        st.image(im.crop((im.width*box[0]/100, im.height*vertical[0]/100, im.width*box[1]/100, im.height*vertical[1]/100)), width='stretch')
                        if item['forensic'].get('scores_path'):
                            from local_models import region_evidence
                            roi = region_evidence(item['forensic'], [box[0]/100,vertical[0]/100,box[1]/100,vertical[1]/100])
                            st.write('**选定屏幕检测：** ' + roi['verdict'])
                            st.caption(f"该区域强响应占比 {roi['high_response_fraction']:.1%}（演示阈值，未经专项校准）")
                            item['screen_region'] = {'box':[box[0]/100,vertical[0]/100,box[1]/100,vertical[1]/100], **roi}
                    with st.form('edit_' + str(i+slot) + item['sha256']):
                        stages = ['阶段待确认', '维修前', '维修后', '辅助材料']
                        stage = st.selectbox('照片阶段', stages, index=stages.index(item['stage']))
                        temperature = st.text_input('温度（无法读取请留空）', str(f.get('temperature')) if f.get('temperature') is not None else '')
                        types = ['unknown', 'air', 'surface', 'plant']
                        measurement = st.selectbox('测量对象', types, index=types.index(f.get('measurement', 'unknown')), format_func=lambda x: {'unknown':'不确定','air':'空气温度','surface':'表面温度','plant':'热力站参数'}[x])
                        date = st.text_input('水印日期 YYYY-MM-DD', f.get('watermark_date') or '')
                        time = st.text_input('水印时间 HH:MM', f.get('watermark_time') or '')
                        location = st.text_input('水印地点', f.get('location') or '')
                        unit = st.selectbox('单位', ['C', 'F', 'unknown'], index=['C','F','unknown'].index(f.get('unit', 'unknown')))
                        clear = st.checkbox('人工确认读数清晰', value=f.get('readability') == 'clear')
                        if st.form_submit_button('保存人工核对'):
                            try:
                                if date: datetime.strptime(date, '%Y-%m-%d')
                                if time: datetime.strptime(time, '%H:%M')
                                item['facts'].update(temperature=float(temperature) if temperature else None, unit=unit, measurement=measurement, watermark_date=date or None, watermark_time=time or None, location=location or None, readability='clear' if clear else 'uncertain')
                                item['stage'] = stage
                                item['source'] = '人工核对'
                                st.rerun()
                            except ValueError:
                                st.error('请检查温度及日期时间格式。')
with tab2:
    st.subheader('待核对线索')
    for warning in warnings:
        st.warning(warning)
    if not warnings:
        st.info('现有规则未发现冲突，不代表照片真实或测量合规。')
    st.caption('未配置合格温度阈值，不自动判断供暖达标；目前未实现可靠的裁剪复用检索。')
with tab3:
    decision = st.radio('审核意见', ['待审核', '通过', '需补拍', '人工核查'], horizontal=True)
    notes = st.text_area('审核说明')
    report = {'case_id': case_id, 'created_at': datetime.now(timezone.utc).isoformat(), 'images': items,
              'review_signals': warnings, 'decision': decision, 'review_notes': notes,
              'limitations': ['规则线索不证明篡改', '未运行模型不得推断真实性', '无法仅凭静态照片证明测量过程合规']}
    st.download_button('下载审核报告 JSON', json.dumps(report, ensure_ascii=False, indent=2), 'review.json', 'application/json')
    if st.button('保存到本地审核历史'):
        save_report(report)
        st.success('已保存。后续导入照片会与本地审核历史比对。')
