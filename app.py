import streamlit as st
import os
import google.generativeai as genai
from rdflib import Graph

# ==========================================
# [설정] 페이지 및 API 설정
# ==========================================
st.set_page_config(page_title="Gemini - 노벨상 제조기", page_icon="🏆", layout="wide")

# [보안] API 키 설정
# st.secrets["GEMINI_API_KEY"] 혹은 아래 변수에 직접 입력
try:
    api_key = st.secrets["AIzaSyDjesITZRyfEAD2SnX799hR0TjAaQAWo7w"]
except:
    api_key = "YOUR_GEMINI_API_KEY_HERE" 

genai.configure(api_key=api_key)

# [수정] 요청하신 모델명 적용 (gemini-pro-3-preview)
# ※ 주의: 해당 모델명이 실제 Google AI Studio에서 유효한지 확인해주세요.
# 만약 에러가 난다면 'gemini-1.5-pro' 또는 'gemini-2.0-flash-exp' 등으로 변경해야 합니다.
model = genai.GenerativeModel('gemini-pro-3-preview')

# ==========================================
# [CSS] 디자인 스타일링 (카드 UI 유지)
# ==========================================
st.markdown("""
<style>
    .stApp { background-color: #f9fafb; }
    
    /* 헤더 스타일 */
    .main-header {
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        padding: 40px 0;
        text-align: center;
    }
    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #191f28;
        margin: 0;
    }
    .sub-title {
        font-size: 18px;
        color: #8b95a1;
        margin-top: 10px;
    }
    
    /* ------------------------------------------- */
    /* [유지] 카드형 UI 디자인 */
    /* ------------------------------------------- */
    .program-card {
        background-color: white;
        border-radius: 20px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        border: 1px solid #f0f0f0;
        transition: transform 0.2s;
        height: 320px; 
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        position: relative;
    }
    .program-card:hover { transform: translateY(-5px); }

    .card-content { flex: 1; }

    .icon-box {
        font-size: 40px;
        position: absolute;
        top: 20px;
        right: 20px;
        filter: drop-shadow(0 2px 4px rgba(0,0,0,0.1));
    }
    
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 600;
        background-color: #f2f4f6;
        color: #4e5968;
        margin-bottom: 10px;
    }
    
    .card-title {
        font-size: 20px;
        font-weight: 700;
        color: #191f28;
        margin-bottom: 8px;
        line-height: 1.4;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
        padding-right: 50px;
    }
    
    .card-desc {
        font-size: 15px;
        color: #4e5968;
        line-height: 1.5;
        margin-top: 10px;
        display: -webkit-box;
        -webkit-line-clamp: 3;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }
    
    .action-btn {
        display: block;
        width: 100%;
        text-align: center;
        background-color: #e8f3ff;
        color: #1b64da;
        text-decoration: none;
        padding: 12px 0;
        border-radius: 12px;
        font-size: 15px;
        font-weight: 600;
        transition: 0.2s;
        margin-top: 15px;
    }
    .action-btn:hover {
        background-color: #3182f6;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# [함수] RAG 데이터 로드
# ==========================================
@st.cache_resource
def load_rag_context():
    context_text = ""
    # 1. TXT 파일 로드
    txt_path = 'data.txt'
    if os.path.exists(txt_path):
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                context_text += f"\n=== [참고 문서 데이터 (TXT)] ===\n{f.read()}\n"
        except Exception as e:
            st.error(f"TXT 파일 로드 중 오류: {e}")
    
    # 2. TTL 파일 로드
    ttl_path = 'ontology.ttl'
    if os.path.exists(ttl_path):
        try:
            g = Graph()
            g.parse(ttl_path, format="turtle")
            ttl_data = g.serialize(format="nt")
            context_text += f"\n=== [온톨로지 구조 데이터 (TTL)] ===\n{ttl_data}\n"
        except Exception as e:
            st.error(f"TTL 파일 로드 중 오류: {e}")

    if not context_text:
        return "데이터 파일이 없습니다. 일반적인 지식으로 답변하세요."
    return context_text

# ==========================================
# [헤더] 문구 변경 (샤모아 -> 노벨상 제조기)
# ==========================================
st.markdown("""
<div class="main-header">
    <div class="main-title">Gemini</div>
    <div class="sub-title">온톨로지 수강생들을 위해 노벨상 제조기를 만들었습니다 🎓</div>
</div>
""", unsafe_allow_html=True)

st.write("---")

# ==========================================
# [메인] 탭 구성
# ==========================================
tab1, tab