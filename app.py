import os
from pathlib import Path
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import pypdf

# Configuração da página do Streamlit
st.set_page_config(
    page_title="Mir - Central Ferroviária",
    page_icon="🚆",
    layout="wide"
)

# Inicializa o estado da escolha do sistema
if "sistema_ativo" not in st.session_state:
    st.session_state.sistema_ativo = None

# --- CSS CUSTOMIZADO ---
st.markdown("""
    <style>
        section[data-testid="stSidebar"] * { font-size: 16px !important; }
        .stChatMessage * { font-size: 17px !important; }
        .stChatInput textarea { font-size: 16px !important; }
        div.stButton, div.stLinkButton { width: 100% !important; }
        .stButton button, div.stLinkButton a {
            width: 100% !important;
            background-color: #ff4b4b !important;
            color: white !important;
            border-radius: 6px !important;
            border: none !important;
            font-weight: 600 !important;
            text-align: center !important;
            padding: 0.5rem 1rem !important;
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
        }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def carregar_componentes_ia():
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    chroma_client = chromadb.PersistentClient(path="./db_docs")
    return encoder, chroma_client

encoder, chroma_client = carregar_componentes_ia()

OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    default_headers={"HTTP-Referer": "http://localhost", "X-Title": "Mir Central"}
)

# --- TELA INICIAL ---
if st.session_state.sistema_ativo is None:
    st.title("🚆 MIR - Plataforma de Inteligência Ferroviária")
    st.markdown("### Selecione qual sistema de suporte técnico ou consulta você deseja acessar:")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🚂 Locomotivas")
        if st.button("🚂 Acessar Locomotivas"):
            st.session_state.sistema_ativo = "locomotivas"
            st.rerun()
        st.subheader("🛤️ Máquinas de Via")
        if st.button("🛤️ Acessar Máquinas de Via"):
            st.session_state.sistema_ativo = "maquinas_via"
            st.rerun()
    with col2:
        st.subheader("📦 Central de Catálogos")
        st.link_button("🔗 Ir para Catálogos e Materiais", "https://mirmaterial.streamlit.app/")

else:
    # --- CONFIGURAÇÃO DE MÓDULOS ---
    if st.session_state.sistema_ativo == "locomotivas":
        PASTA_BASE_MANUAIS = Path("./Docs_Locomotivas")
        collection_name = "mir_suporte_locomotivas"
        titulo_sidebar = "🚆 MIR - Locomotivas"
        msg_inicial_duvida = "Olá. Use este canal para retirar dúvidas gerais sobre locomotivas."
        msg_inicial_falha = "Olá. Descreva a falha observada na locomotiva e o modelo."
    else:
        PASTA_BASE_MANUAIS = Path("./Docs_MaquinasVia")
        collection_name = "mir_suporte_via"
        titulo_sidebar = "🛤️ MIR - Máquinas de Via"
        msg_inicial_duvida = "Olá. Use este canal para retirar dúvidas sobre máquinas de via."
        msg_inicial_falha = "Olá. Descreva a falha na máquina de via."

    PASTA_BASE_MANUAIS.mkdir(exist_ok=True)
    collection = chroma_client.get_or_create_collection(name=collection_name)

    # --- FUNÇÃO INDEXAR ---
    def indexar_arquivo(caminho_arquivo):
        try:
            nome_arq = caminho_arquivo.name
            collection.delete(where={"source": nome_arq})
            reader = pypdf.PdfReader(str(caminho_arquivo))
            for i, page in enumerate(reader.pages):
                texto = page.extract_text()
                if texto and texto.strip():
                    collection.upsert(
                        ids=[f"{nome_arq}_p{i}"],
                        embeddings=[encoder.encode([texto]).tolist()[0]],
                        documents=[texto],
                        metadatas=[{"source": nome_arq, "page": i}]
                    )
            return True
        except Exception: return False

    # --- GESTÃO DE ESTADO (CORRIGIDO) ---
    chave_chats_duvidas = f"chats_duvidas_{st.session_state.sistema_ativo}"
    chave_chat_atual_duvidas = f"chat_atual_duvidas_{st.session_state.sistema_ativo}"
    chave_chats_falhas = f"chats_falhas_{st.session_state.sistema_ativo}"
    chave_chat_atual_falhas = f"chat_atual_falhas_{st.session_state.sistema_ativo}"

    if chave_chats_duvidas not in st.session_state:
        st.session_state[chave_chats_duvidas] = [{"id": 1, "titulo": "Nova Dúvida", "mensagens": [{"role": "assistant", "content": msg_inicial_duvida}]}]
    if chave_chat_atual_duvidas not in st.session_state:
        st.session_state[chave_chat_atual_duvidas] = 0

    if chave_chats_falhas not in st.session_state:
        st.session_state[chave_chats_falhas] = [{"id": 1, "titulo": "Nova Falha", "mensagens": [{"role": "assistant", "content": msg_inicial_falha}]}]
    if chave_chat_atual_falhas not in st.session_state:
        st.session_state[chave_chat_atual_falhas] = 0

    # --- SIDEBAR E LÓGICA DE CHAT ---
    with st.sidebar:
        if st.button("🏠 Voltar ao Menu Principal"):
            st.session_state.sistema_ativo = None
            st.rerun()
        # ... (O restante da sua lógica de sidebar permanece a mesma, apenas certifique-se de manter a indentação de 4 espaços)