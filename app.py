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

# --- CSS CUSTOMIZADO PARA PADRONIZAÇÃO VISUAL ---
st.markdown("""
    <style>
        section[data-testid="stSidebar"] * {
            font-size: 16px !important;
        }
        .stChatMessage * {
            font-size: 17px !important;
        }
        .stChatInput textarea {
            font-size: 16px !important;
        }
        div.stButton > button {
            background-color: #ff4b4b !important;
            color: white !important;
            border-radius: 6px !important;
            border: none !important;
            font-weight: 600 !important;
            width: 100% !important;
        }
        div.stButton > button:hover {
            background-color: #ff2b2b !important;
            color: white !important;
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

# --- TELA INICIAL: ESCOLHA DE SISTEMA ---
if st.session_state.sistema_ativo is None:
    st.title("🚆 MIR - Plataforma de Inteligência Ferroviária")
    st.markdown("### Selecione qual sistema de suporte técnico ou consulta você deseja acessar:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("---")
        st.subheader("🚂 Locomotivas")
        st.markdown("Suporte especializado em locomotivas, sistemas de freio CCBII e elétrica ferroviária.")
        if st.button("🚂 Acessar Locomotivas", key="btn_loc"):
            st.session_state.sistema_ativo = "locomotivas"
            st.rerun()
            
        st.markdown("---")
        st.subheader("🛤️ Máquinas de Via")
        st.markdown("Suporte especializado em máquinas de via permanente, socadoras (Plasser) e hidráulica.")
        if st.button("🛤️ Acessar Máquinas de Via", key="btn_via"):
            st.session_state.sistema_ativo = "maquinas_via"
            st.rerun()        
            
    with col2:
        st.markdown("---")
        st.subheader("🔧 Especificação de Mangueiras & Adaptadores")
        st.markdown("Painel interativo para mangueiras (2781, 1503, FC350-) e adaptadores com tabelas JIC e NPT.")
        if st.button("🔧 Acessar Mangueiras e Adaptadores", key="btn_mangueiras"):
            st.session_state.sistema_ativo = "mangueiras"
            st.rerun()

        st.markdown("---")
        st.subheader("📦 Central de Catálogos e Materiais")
        st.markdown("Acesse o sistema externo de consulta de catálogos em PDF (Locomotivas e Máquinas) e códigos internos.")
        st.link_button(
            "🔗 Ir para Catálogos e Materiais", 
            "https://mirmaterial.streamlit.app/"
        )

# --- MÓDULOS DE SUPORTE TÉCNICO ---
else:
    if st.session_state.sistema_ativo == "mangueiras":
        with st.sidebar:
            if st.button("🏠 Voltar ao Menu Principal"):
                st.session_state.sistema_ativo = None
                st.rerun()
            st.markdown("<br>", unsafe_allow_html=True)
            st.link_button("📦 Catálogos e Códigos", "https://mirmaterial.streamlit.app/")
            st.divider()
            st.markdown("## 🔧 MIR - Hidráulica")
            st.caption("Mangueiras & Adaptadores JIC/NPT")

        st.title("🔧 Painel de Especificação de Mangueiras e Adaptadores")
        
        aba_mangueira, aba_adaptador = st.tabs(["🪡 Mangueiras Hidráulicas", "⚙️ Adaptadores e Conexões"])

        # TAB 1: MANGUEIRAS COM BUSCA MANUAL POR PAQUÍMETRO
        with aba_mangueira:
            st.markdown("### Especificação de Mangueiras (Modelos 2781, 1503, FC350-)")
            st.markdown("💡 *Dica:* Insira as medidas obtidas com o paquímetro para que o sistema encontre o modelo e tamanho tabelado mais próximo automaticamente.")

            tabelas_modelos = {
                "2781-": {
                    "descricao": "Modelo 2781 (Alta Pressão / Robusto)",
                    "dados": [
                        {"tamanho": "4", "ext": 17.5, "int": 6.4, "pressao": "400bar"},
                        {"tamanho": "6", "ext": 21.4, "int": 9.5, "pressao": "350bar"},
                        {"tamanho": "8", "ext": 24.6, "int": 12.7, "pressao": "300bar"},
                        {"tamanho": "10", "ext": 27.8, "int": 15.9, "pressao": "250bar"},
                        {"tamanho": "12", "ext": 31.8, "int": 19.0, "pressao": "225bar"},
                        {"tamanho": "16", "ext": 39.7, "int": 25.4, "pressao": "175bar"},
                        {"tamanho": "20", "ext": 50.8, "int": 31.8, "pressao": "165bar"},
                        {"tamanho": "24", "ext": 57.2, "int": 38.1, "pressao": "135bar"}
                    ]
                },
                "1503-": {
                    "descricao": "Modelo 1503 (Pressão Média/Intermediária)",
                    "dados": [
                        {"tamanho": "6", "ext": 17.0, "int": 8.0, "pressao": "160bar"},
                        {"tamanho": "8", "ext": 19.5, "int": 10.3, "pressao": "140bar"},
                        {"tamanho": "10", "ext": 23.5, "int": 12.7, "pressao": "125bar"},
                        {"tamanho": "12", "ext": 27.4, "int": 15.9, "pressao": "105bar"},
                        {"tamanho": "16", "ext": 31.3, "int": 22.2, "pressao": "55bar"},
                        {"tamanho": "20", "ext": 38.0, "int": 28.6, "pressao": "40bar"},
                        {"tamanho": "24", "ext": 44.5, "int": 35.0, "pressao": "35bar"}
                    ]
                },
                "FC350-": {
                    "descricao": "Modelo FC350- (Baixa/Média Pressão)",
                    "dados": [
                        {"tamanho": "4", "ext": 13.1, "int": 4.8, "pressao": "105bar"},
                        {"tamanho": "5", "ext": 14.7, "int": 6.4, "pressao": "105bar"},
                        {"tamanho": "6", "ext": 17.0, "int": 7.9, "pressao": "105bar"},
                        {"tamanho": "8", "ext": 19.5, "int": 10.3, "pressao": "85bar"},
                        {"tamanho": "10", "ext": 23.4, "int": 12.7, "pressao": "85bar"},
                        {"tamanho": "12", "ext": 27.4, "int": 15.9, "pressao": "50bar"}
                    ]
                }
            }

            with st.form("form_montagem_mangueira"):
                st.markdown("#### 📏 Medições com Paquímetro (Entrada Manual)")
                col_paq1, col_paq2 = st.columns(2)
                with col_paq1:
                    medida_ext_input = st.number_input("Diâmetro Externo Medido (mm)", min_value=0.0, max_value=100.0, value=21.4, step=0.1)
                with col_paq2:
                    medida_int_input = st.number_input("Diâmetro Interno Medido (mm)", min_value=0.0, max_value=100.0, value=9.5, step=0.1)

                st.divider()
                st.markdown("#### 🛠️ Configuração de Terminais e Comprimento")
                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    tipo_terminal_1 = st.selectbox("Terminal Lado A", ["Fêmea Giratória JIC", "Macho Fixo NPT", "Flange Code 61", "Flange Code 62", "Olhal Hidráulico"])
                with col_m2:
                    tipo_terminal_2 = st.selectbox("Terminal Lado B", ["Fêmea Giratória JIC", "Macho Fixo NPT", "Flange Code 61", "Flange Code 62", "Olhal Hidráulico"])

                comprimento_mm = st.number_input("Comprimento Total da Mangueira (mm)", min_value=100, max_value=20000, value=1000, step=50)
                observacoes_extras = st.text_area("Observações Específicas (opcional)", placeholder="Ex: Capa protetora contra abrasão...")

                submitted_mangueira = st.form_submit_button("Buscar Correspondência e Gerar Especificação")

            if submitted_mangueira:
                # Algoritmo para encontrar a melhor correspondência (menor diferença absoluta combinada ext + int)
                melhor_match = None
                menor_diferenca = float('inf')

                for mod_key, mod_info in tabelas_modelos.items():
                    for item in mod_info["dados"]:
                        diff = abs(item["ext"] - medida_ext_input) + abs(item["int"] - medida_int_input)
                        if diff < menor_diferenca:
                            menor_diferenca = diff
                            melhor_match = {
                                "modelo": mod_key,
                                "descricao": mod_info["descricao"],
                                "tamanho": item["tamanho"],
                                "ext_tabelado": item["ext"],
                                "int_tabelado": item["int"],
                                "pressao": item["pressao"]
                            }

                if melhor_match:
                    st.success("Correspondência encontrada com base nos valores informados!")
                    st.info(f"""
                    * **Modelo Identificado:** {melhor_match['modelo']} ({melhor_match['descricao']})
                    * **Tamanho Código Recomendado:** {melhor_match['tamanho']}
                    * **Diâmetro Externo Tabelado:** {melhor_match['ext_tabelado']} mm *(Medido: {medida_ext_input} mm)*
                    * **Diâmetro Interno Tabelado:** {melhor_match['int_tabelado']} mm *(Medido: {medida_int_input} mm)*
                    * **Pressão Operacional:** {melhor_match['pressao']}
                    * **Terminais:** {tipo_terminal_1} / {tipo_terminal_2}
                    * **Comprimento:** {comprimento_mm} mm
                    * **Observações:** {observacoes_extras if observacoes_extras else 'Nenhuma'}
                    """)
                    codigo_gerado = f"{melhor_match['modelo']}{melhor_match['tamanho']}-{comprimento_mm}MM"
                    st.markdown(f"**Código de Referência Sugerido:** `{codigo_gerado}`")
                else:
                    st.warning("Nenhum modelo compatível encontrado com margem próxima.")

        # TAB 2: ADAPTADORES (JIC / NPT)
        with aba_adaptador:
            st.markdown("### Configuração de Adaptadores e Conexões (Padrões JIC e NPT)")
            
            tabela_jic = [
                {"ng": "4", "ext_d": 11.07, "rosca": "JIC 7/16-20"},
                {"ng": "6", "ext_d": 14.25, "rosca": "JIC 9/16-18"},
                {"ng": "8", "ext_d": 19.00, "rosca": "JIC 3/4-16"},
                {"ng": "10", "ext_d": 22.17, "rosca": "JIC 7/8-14"},
                {"ng": "12", "ext_d": 26.95, "rosca": "JIC 1 1/16-12"},
                {"ng": "16", "ext_d": 33.30, "rosca": "JIC 1 5/16-12"},
                {"ng": "20", "ext_d": 41.22, "rosca": "JIC 1 9/16-18"},
                {"ng": "24", "ext_d": 47.57, "rosca": "JIC 1 7/8-12"},
                {"ng": "32", "ext_d": 63.45, "rosca": "JIC 2 1/2-12"}
            ]

            tabela_npt = [
                {"ng": "2", "ext_d": "N/A", "rosca": "NPT 1/8\"-27"},
                {"ng": "4", "ext_d": 13.50, "rosca": "NPT 1/4\"-18"},
                {"ng": "6", "ext_d": 16.50, "rosca": "NPT 3/8\"-18"},
                {"ng": "8", "ext_d": 21.00, "rosca": "NPT 1/2\"-14"},
                {"ng": "12", "ext_d": 26.00, "rosca": "NPT 3/4\"-14"},
                {"ng": "16", "ext_d": 33.00, "rosca": "NPT 1\"-11 1/2"},
                {"ng": "20", "ext_d": 41.00, "rosca": "NPT 1 1/4\"-11 1/2"},
                {"ng": "24", "ext_d": 47.00, "rosca": "NPT 1 1/2\"-11 1/2"},
                {"ng": "32", "ext_d": 59.00, "rosca": "NPT 2\"-11 1/2"}
            ]

            modelos_adaptadores = [
                "2021- (Adaptador Macho Giratório 90°)",
                "2041- (Adaptador Reto Macho-Macho)",
                "2042- (Adaptador Macho 45°)",
                "2043- (Adaptador Macho 90°)",
                "2023- (Adaptador Fêmea Giratória 90°)",
                "2024- (Adaptador Fêmea Giratória 45°)",
                "2081- (Adaptador Tê / Redução)",
                "2088- (Adaptador Cruz / Tê Especial)",
                "2030- (Tampão Cego / Plugue)",
                "2092- (União Anteparo / Conexão Reta)"
            ]

            with st.form("form_montagem_adaptador"):
                modelo_adaptador = st.selectbox("Selecione o Modelo do Adaptador", modelos_adaptadores)
                
                col_ad1, col_ad2 = st.columns(2)
                
                with col_ad1:
                    st.markdown("#### Rosca Extremidade 1")
                    tipo_rosca_1 = st.selectbox("Padrão Lado 1", ["JIC", "NPT"], key="t_rosca_1")
                    if tipo_rosca_1 == "JIC":
                        opcoes_jic_1 = [f"NG {item['ng']} - {item['rosca']} (Ext: {item['ext_d']}mm)" for item in tabela_jic]
                        escolha_l1 = st.selectbox("Medida Lado 1 (JIC)", opcoes_jic_1, key="es_jic_1")
                    else:
                        opcoes_npt_1 = [f"NG {item['ng']} - {item['rosca']} (Ext: {item['ext_d']}mm)" for item in tabela_npt]
                        escolha_l1 = st.selectbox("Medida Lado 1 (NPT)", opcoes_npt_1, key="es_npt_1")

                with col_ad2:
                    st.markdown("#### Rosca Extremidade 2")
                    tipo_rosca_2 = st.selectbox("Padrão Lado 2", ["JIC", "NPT"], key="t_rosca_2")
                    if tipo_rosca_2 == "JIC":
                        opcoes_jic_2 = [f"NG {item['ng']} - {item['rosca']} (Ext: {item['ext_d']}mm)" for item in tabela_jic]
                        escolha_l2 = st.selectbox("Medida Lado 2 (JIC)", opcoes_jic_2, key="es_jic_2")
                    else:
                        opcoes_npt_2 = [f"NG {item['ng']} - {item['rosca']} (Ext: {item['ext_d']}mm)" for item in tabela_npt]
                        escolha_l2 = st.selectbox("Medida Lado 2 (NPT)", opcoes_npt_2, key="es_npt_2")

                qtd_pecas = st.number_input("Quantidade", min_value=1, max_value=100, value=1)
                obs_adaptador = st.text_area("Observações do Adaptador (opcional)", placeholder="Ex: Material aço carbono zincado...")

                submitted_adaptador = st.form_submit_button("Gerar Código do Adaptador")

            if submitted_adaptador:
                prefixo_codigo = modelo_adaptador.split(" ")[0]
                st.success("Adaptador especificado com sucesso!")
                st.info(f"""
                * **Modelo do Adaptador:** {modelo_adaptador}
                * **Lado 1:** {escolha_l1}
                * **Lado 2:** {escolha_l2}
                * **Quantidade:** {qtd_pecas} unidade(s)
                * **Observações:** {obs_adaptador if obs_adaptador else 'Nenhuma'}
                """)
                
                codigo_adaptador_final = f"{prefixo_codigo} [L1: {escolha_l1}] x [L2: {escolha_l2}]"
                st.markdown(f"**Código de Referência do Adaptador:** `{codigo_adaptador_final}`")

    else:
        # Lógica padrão de Locomotivas e Máquinas de Via
        if st.session_state.sistema_ativo == "locomotivas":
            PASTA_BASE_MANUAIS = Path("./Docs_Locomotivas")
            collection_name = "mir_suporte_locomotivas"
            titulo_sidebar = "🚆 MIR - Locomotivas"
            msg_inicial_duvida = "Olá. Use este canal para retirar dúvidas gerais sobre locomotivas."
            msg_inicial_falha = "Olá. Descreva a falha observada na locomotiva e o modelo (ex: AC44)."
        else:
            PASTA_BASE_MANUAIS = Path("./Docs_MaquinasVia")
            collection_name = "mir_suporte_via"
            titulo_sidebar = "🛤️ MIR - Máquinas de Via"
            msg_inicial_duvida = "Olá. Use este canal para retirar dúvidas sobre máquinas de via."
            msg_inicial_falha = "Olá. Descreva a falha na máquina de via (ex: Socadora Plasser)."

        PASTA_BASE_MANUAIS.mkdir(exist_ok=True)
        collection = chroma_client.get_or_create_collection(name=collection_name)

        def indexar_arquivo(caminho_arquivo):
            try:
                nome_arq = caminho_arquivo.name
                existing = collection.get(where={"source": nome_arq})
                if existing and len(existing.get('ids', [])) > 0:
                    collection.delete(where={"source": nome_arq})
                
                reader = pypdf.PdfReader(str(caminho_arquivo))
                total_paginas = len(reader.pages)
                if total_paginas == 0:
                    return False

                for i, page in enumerate(reader.pages):
                    texto = page.extract_text()
                    if texto and texto.strip():
                        page_id = f"{nome_arq}_p{i}"
                        embedding = encoder.encode([texto]).tolist()
                        collection.upsert(
                            ids=[page_id],
                            embeddings=embedding,
                            documents=[texto],
                            metadatas=[{"source": nome_arq, "page": i}]
                        )
                return True
            except Exception as e:
                print(f"Erro ao indexar {caminho_arquivo}: {e}")
                return False

        chave_chats_duvidas = f"chats_duvidas_{st.session_state.sistema_ativo}"
        chave_chat_atual_duvidas = f"chat_atual_duvidas_{st.session_state.sistema_ativo}"
        chave_chats_falhas = f"chats_falhas_{st.session_state.sistema_ativo}"
        chave_chat_atual_falhas = f"chat_atual_falhas_{st.session_state.sistema_ativo}"

        if chave_chats_duvidas not in st.session_state:
            st.session_state[chave_chats_duvidas] = [{"id": 1, "titulo": "Nova Dúvida", "mensagens": [{"role": "assistant", "content": msg_inicial_duvida}]}]
        if chave_chat_atual_duvidas not in st.session_state:
            st.session_state[chave_chat_atual_duvidas] = 0

        if chave_chats_falhas not in st.session_state:
            st.session_state[chave_chats_falhas] = [{"id": 1, "titulo": "Nova Análise", "mensagens": [{"role": "assistant", "content": msg_inicial_falha}]}]
        if chave_chat_atual_falhas not in st.session_state:
            st.session_state[chave_chat_atual_falhas] = 0

        with st.sidebar:
            if st.button("🏠 Voltar ao Menu Principal"):
                st.session_state.sistema_ativo = None
                st.rerun()

            st.markdown("<br>", unsafe_allow_html=True)
            st.link_button("📦 Catálogos e Códigos", "https://mirmaterial.streamlit.app/")

            st.divider()
            st.markdown(f"## {titulo_sidebar}")
            st.caption("Inteligência em Manutenção")
            st.divider()

            aba_selecionada = st.radio(
                "Navegação",
                ["📖 Dúvidas Técnicas", "⚙️ Análise / Consulta", "📂 Adicionar Conhecimento"],
                label_visibility="collapsed"
            )

            st.divider()
            
            if aba_selecionada == "📖 Dúvidas Técnicas":
                st.markdown("### 🕒 Histórico de Dúvidas")
                if st.button("➕ Novo Chat de Dúvidas"):
                    lista_chats = st.session_state[chave_chats_duvidas]
                    novo_id = len(lista_chats) + 1
                    lista_chats.append({
                        "id": novo_id, 
                        "titulo": f"Dúvida {novo_id}", 
                        "mensagens": [{"role": "assistant", "content": msg_inicial_duvida}]
                    })
                    st.session_state[chave_chat_atual_duvidas] = len(lista_chats) - 1
                    st.rerun()

                for i, chat in enumerate(st.session_state[chave_chats_duvidas][-10:]):
                    col_h1, col_h2 = st.columns([0.8, 0.2])
                    if col_h1.button(chat["titulo"], key=f"btn_duvida_{i}"):
                        st.session_state[chave_chat_atual_duvidas] = i
                        st.rerun()
                    if col_h2.button("🗑️", key=f"del_duvida_{i}"):
                        lista_chats = st.session_state[chave_chats_duvidas]
                        if len(lista_chats) > 1:
                            del lista_chats[i]
                            st.session_state[chave_chat_atual_duvidas] = max(0, i - 1)
                            st.rerun()
                        else:
                            st.warning("Mínimo de 1 chat.")

            elif aba_selecionada == "⚙️ Análise / Consulta":
                st.markdown("### 🕒 Histórico de Consultas")
                if st.button("➕ Novo Chat de Consulta"):
                    lista_chats = st.session_state[chave_chats_falhas]
                    novo_id = len(lista_chats) + 1
                    lista_chats.append({
                        "id": novo_id, 
                        "titulo": f"Consulta {novo_id}", 
                        "mensagens": [{"role": "assistant", "content": msg_inicial_falha}]
                    })
                    st.session_state[chave_chat_atual_falhas] = len(lista_chats) - 1
                    st.rerun()

                for i, chat in enumerate(st.session_state[chave_chats_falhas][-10:]):
                    col_h1, col_h2 = st.columns([0.8, 0.2])
                    if col_h1.button(chat["titulo"], key=f"btn_falha_{i}"):
                        st.session_state[chave_chat_atual_falhas] = i
                        st.rerun()
                    if col_h2.button("🗑️", key=f"del_falha_{i}"):
                        lista_chats = st.session_state[chave_chats_falhas]
                        if len(lista_chats) > 1:
                            del lista_chats[i]
                            st.session_state[chave_chat_atual_falhas] = max(0, i - 1)
                            st.rerun()
                        else:
                            st.warning("Mínimo de 1 chat.")

            st.divider()
            total_pdfs = len(list(PASTA_BASE_MANUAIS.glob("**/*.pdf")))
            try:
                total_trechos = collection.count()
            except Exception:
                total_trechos = 0
                
            st.info(f"**Status:** Sistema Pronto\n\n📁 {total_pdfs} PDFs encontrados\n📚 {total_trechos} trechos ativos")

        if aba_selecionada == "📖 Dúvidas Técnicas":
            st.markdown(f"### 📖 Dúvidas Técnicas - {st.session_state.sistema_ativo.capitalize()}")
            
            chat_idx = st.session_state[chave_chat_atual_duvidas]
            chat_atual = st.session_state[chave_chats_duvidas][chat_idx]

            for msg in chat_atual["mensagens"]:
                with st.chat_message("assistant" if msg["role"] == "assistant" else "user", avatar="🔹" if msg["role"] == "assistant" else "👤"):
                    st.markdown(msg["content"])

            pergunta = st.chat_input("Digite sua dúvida técnica...", key="input_duvida")

            if pergunta:
                chat_atual["mensagens"].append({"role": "user", "content": pergunta})
                
                if chat_atual["titulo"].startswith("Nova Dúvida") or chat_atual["titulo"].startswith("Dúvida"):
                    chat_atual["titulo"] = pergunta[:25] + "..." if len(pergunta) > 25 else pergunta

                with st.chat_message("user", avatar="👤"):
                    st.markdown(pergunta)

                with st.chat_message("assistant", avatar="🔹"):
                    with st.spinner("Realizando busca profunda na base de dados..."):
                        try:
                            pergunta_vetor = encoder.encode([pergunta]).tolist()
                            resultados = collection.query(query_embeddings=pergunta_vetor, n_results=12)
                            
                            contexto_partes = []
                            if resultados and resultados["documents"] and resultados["documents"][0]:
                                for doc, meta in zip(resultados["documents"][0], resultados["metadatas"][0]):
                                    contexto_partes.append(f"Fonte: {meta.get('source', 'Desconhecido')} | Conteúdo: {doc}")
                            
                            contexto = "\n\n".join(contexto_partes) if contexto_partes else "Nenhum trecho correspondente encontrado."
                            
                            system_prompt = (
                                "Você é o Mir, um assistente especialista sênior em engenharia e manutenção ferroviária. "
                                "Sua função principal é atuar como um consultor técnico de suporte, respondendo a dúvidas, "
                                "explicando conceitos e detalhando especificações com base estrita nos manuais e documentos indexados.\n\n"
                                f"CONTEXTO TÉCNICO EXTRAÍDO DOS MANUAIS:\n{contexto}"
                            )

                            messages_payload = [{"role": "system", "content": system_prompt}]
                            for m in chat_atual["mensagens"][:-1]:
                                messages_payload.append({"role": m["role"], "content": m["content"]})
                            
                            messages_payload.append({"role": "user", "content": chat_atual["mensagens"][-1]["content"]})

                            response = client.chat.completions.create(
                                model="openai/gpt-4o-mini",
                                messages=messages_payload,
                                temperature=0.0
                            )
                            
                            resposta_ia = response.choices[0].message.content
                            st.markdown(resposta_ia)
                            chat_atual["mensagens"].append({"role": "assistant", "content": resposta_ia})
                            st.rerun()
                            
                        except Exception as e:
                            erro_msg = f"Erro na consulta: {e}"
                            st.error(erro_msg)
                            chat_atual["mensagens"].append({"role": "assistant", "content": erro_msg})

        elif aba_selecionada == "⚙️ Análise / Consulta":
            st.markdown(f"### ⚙️ Análise Técnica - {st.session_state.sistema_ativo.capitalize()}")
            
            chat_idx = st.session_state[chave_chat_atual_falhas]
            chat_atual = st.session_state[chave_chats_falhas][chat_idx]

            for msg in chat_atual["mensagens"]:
                with st.chat_message("assistant" if msg["role"] == "assistant" else "user", avatar="🔹" if msg["role"] == "assistant" else "👤"):
                    st.markdown(msg["content"])

            pergunta = st.chat_input("Insira os parâmetros ou dados para análise...", key="input_falha")

            if pergunta:
                chat_atual["mensagens"].append({"role": "user", "content": pergunta})
                
                if chat_atual["titulo"].startswith("Nova Análise") or chat_atual["titulo"].startswith("Consulta"):
                    chat_atual["titulo"] = pergunta[:25] + "..." if len(pergunta) > 25 else pergunta

                with st.chat_message("user", avatar="👤"):
                    st.markdown(pergunta)

                with st.chat_message("assistant", avatar="🔹"):
                    with st.spinner("Analisando especificações e normas..."):
                        try:
                            pergunta_vetor = encoder.encode([pergunta]).tolist()
                            resultados = collection.query(query_embeddings=pergunta_vetor, n_results=6)
                            
                            contexto_partes = []
                            fontes_encontradas = set()
                            if resultados and resultados["documents"] and resultados["documents"][0]:
                                docs = resultados["documents"][0]
                                metas = resultados["metadatas"][0] if resultados.get("metadatas") else [{}] * len(docs)
                                for doc, meta in zip(docs, metas):
                                    contexto_partes.append(doc)
                                    if meta and "source" in meta:
                                        fontes_encontradas.add(meta["source"])
                            
                            contexto = "\n\n".join(contexto_partes) if contexto_partes else "Nenhum trecho correspondente encontrado na base local."
                            fontes_str = ", ".join(fontes_encontradas) if fontes_encontradas else "Nenhum manual PDF local indexado para citação."

                            system_prompt = (
                                "You are Mir, um engenheiro especialista sênior em manutenção ferroviária.\n"
                                f"Contexto técnico extraído dos manuais locais:\n{contexto}\n\n"
                                f"Fontes/Documentos disponíveis para referência: {fontes_str}"
                            )

                            messages_payload = [{"role": "system", "content": system_prompt}]
                            for m in chat_atual["mensagens"][:-1]:
                                messages_payload.append({"role": m["role"], "content": m["content"]})
                            
                            messages_payload.append({"role": "user", "content": chat_atual["mensagens"][-1]["content"]})

                            response = client.chat.completions.create(
                                model="openai/gpt-4o-mini",
                                messages=messages_payload,
                                temperature=0.1
                            )
                            resposta_ia = response.choices[0].message.content
                            st.markdown(resposta_ia)
                            chat_atual["mensagens"].append({"role": "assistant", "content": resposta_ia})
                            st.rerun()
                        except Exception as e:
                            erro_msg = f"Erro no pipeline de IA: {e}"
                            st.error(erro_msg)
                            chat_atual["mensagens"].append({"role": "assistant", "content": erro_msg})

        elif aba_selecionada == "📂 Adicionar Conhecimento":
            st.markdown(f"# 📂 Adicionar Conhecimento - {st.session_state.sistema_ativo.capitalize()}")
            st.markdown("Carregue novos manuais ou especificações em PDF para expandir imediatamente a base.")

            uploaded_files = st.file_uploader(
                "Carregar documentos técnicos (PDF)", type=["pdf"], accept_multiple_files=True
            )

            if uploaded_files:
                for uploaded_file in uploaded_files:
                    caminho_salvamento = PASTA_BASE_MANUAIS / uploaded_file.name
                    with open(caminho_salvamento, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                st.success(f"{len(uploaded_files)} arquivo(s) salvo(s) com sucesso!")

            st.divider()

            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("🧠 Indexar Novos PDFs da Pasta"):
                    arquivos = list(PASTA_BASE_MANUAIS.glob("**/*.pdf"))
                    if not arquivos:
                        st.warning(f"Nenhum arquivo PDF encontrado em {PASTA_BASE_MANUAIS}.")
                    else:
                        barra = st.progress(0, text="Iniciando indexação...")
                        total_arq = len(arquivos)
                        for idx, arq in enumerate(arquivos):
                            indexar_arquivo(arq)
                            barra.progress((idx + 1) / total_arq, text=f"Indexando: {arq.name}")
                        st.success("Aprendizado concluído com sucesso!")
                        st.rerun()

            with col_btn2:
                if st.button("🗑️ Limpar Base de Dados Indexada"):
                    try:
                        chroma_client.delete_collection(collection_name)
                        chroma_client.get_or_create_collection(name=collection_name)
                        st.success("Base de dados limpa com sucesso!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Erro ao limpar base: {e}")