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
    api_key = st.secrets["GEMINI_API_KEY"]
except:
    api_key = "YOUR_GEMINI_API_KEY_HERE" 

genai.configure(api_key=api_key)

# [수정] 요청하신 모델명 적용 (gemini-pro-3-preview)
# ※ 주의: 해당 모델명이 실제 Google AI Studio에서 유효한지 확인해주세요.
# 만약 에러가 난다면 'gemini-1.5-pro' 또는 'gemini-2.0-flash-exp' 등으로 변경해야 합니다. gemini-3-pro-preview
model = genai.GenerativeModel('gemini-2.0-flash-exp')

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
tab1, tab2 = st.tabs(["📘 연구 주제(예시)", "🤖 노벨상 제조기 (Chat)"])

# ----------------------------------------------------------------
# [Tab 1] 카드형 UI (문구만 변경하여 유지)
# ----------------------------------------------------------------
with tab1:
    # 기존 '샤모아' 소개 문구 대신 노벨상 관련 문구로 대체
    st.markdown("""
    <div style="text-align: center; margin-bottom: 30px;">
        <h3 style="color: #333;">💡 노벨상급 연구 아이디어 예시</h3>
        <p style="color: #666;">온톨로지 구조 분석을 통해 도출된 혁신적인 연구 주제들입니다.</p>
    </div>
    """, unsafe_allow_html=True)

    # [예시 데이터] 카드를 보여주기 위한 더미 데이터 (코드는 필요 없으므로 텍스트만 변경)
    example_projects = [
        {
            "category": "Physics",
            "title": "양자 얽힘과 온톨로지 위상학",
            "desc": "복잡한 양자 상태를 지식 그래프로 모델링하여 새로운 물리 법칙의 가능성을 탐구합니다.",
            "icon": "⚛️",
            "link": "#"
        },
        {
            "category": "Literature",
            "title": "데이터로 읽는 노벨 문학상 수상작",
            "desc": "역대 수상작의 서사 구조와 은유 패턴을 분석하여 수상 가능성이 높은 문학적 코드를 발견합니다.",
            "icon": "📚",
            "link": "#"
        },
        {
            "category": "Medicine",
            "title": "유전자 편집 기술의 윤리적 온톨로지",
            "desc": "CRISPR 기술 발전 시나리오와 생명 윤리 간의 관계를 체계화하여 미래 의료 가이드라인을 제시합니다.",
            "icon": "🧬",
            "link": "#"
        },
        {
            "category": "Peace",
            "title": "글로벌 분쟁 해결을 위한 AI 모델",
            "desc": "국가 간 이해관계 데이터를 온톨로지로 구축하여 지속 가능한 평화 솔루션을 제안합니다.",
            "icon": "🕊️",
            "link": "#"
        }
    ]

    # [카드 렌더링 로직]
    for i in range(0, len(example_projects), 2):
        cols = st.columns(2)
        batch = example_projects[i : i+2]
        
        for idx, item in enumerate(batch):
            with cols[idx]:
                card_html = f"""
                <div class="program-card">
                    <div class="icon-box">{item['icon']}</div>
                    <div class="card-content">
                        <span class="badge">{item['category']}</span>
                        <div class="card-title">{item['title']}</div>
                        <div class="card-desc">{item['desc']}</div>
                    </div>
                    <a href="{item['link']}" class="action-btn">
                        아이디어 상세 보기
                    </a>
                </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)


# ----------------------------------------------------------------
# [Tab 2] AI 챗봇 (RAG)
# ----------------------------------------------------------------
with tab2:
    st.markdown("### 🤖 무엇이든 물어보세요!")
    st.caption("업로드된 TXT 문서와 온톨로지(TTL) 지식을 기반으로 답변합니다.")

    # RAG 데이터 로드
    rag_context = load_rag_context()

    # 시스템 프롬프트
    SYSTEM_PROMPT = f"""
    당신은 '온톨로지 수업 수강생'들을 위한 '노벨상 아이디어 제조기' AI입니다.
    
    [지식 베이스]
    {rag_context}
    
    [행동 지침]
    1. 사용자의 질문에 대해 위 [지식 베이스]를 최우선으로 참고하여 답변하세요.
    2. 데이터에 없는 내용은 일반 지식을 활용하되 구분해서 말해주세요.
    3. 창의적이고 학구적인 '연구 파트너' 톤으로 답변하세요.
    """

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "안녕하세요! 어떤 분야의 노벨상에 도전하고 싶으신가요? 온톨로지 지식으로 도와드릴게요."}
        ]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("질문을 입력하세요..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            try:
                final_input = f"{SYSTEM_PROMPT}\n\n사용자 질문: {prompt}"
                response = model.generate_content(final_input, stream=True)
                
                for chunk in response:
                    if chunk.text:
                        full_response += chunk.text
                        message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
            except Exception as e:
                st.error(f"오류: {e}")
                full_response = "죄송합니다. 오류가 발생했습니다."
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})
