import os
from pathlib import Path
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import pypdf
import pandas as pd

# Configuração da página do Streamlit
st.set_page_config(
    page_title="Mir - Central Ferroviária",
    page_icon="🚆",
    layout="wide"
)

# Inicializa o estado da escolha do sistema
if "sistema_ativo" not in st.session_state:
    st.session_state.sistema_ativo = None

# --- CSS CUSTOMIZADO PARA AUMENTAR AS FONTES ---
st.markdown("""
    <style>
        /* Aumenta a fonte do menu lateral (sidebar) */
        section[data-testid="stSidebar"] * {
            font-size: 16px !important;
        }
        
        /* Aumenta a fonte das mensagens do chat (tanto usuário quanto assistente) */
        .stChatMessage * {
            font-size: 17px !important;
        }
        
        /* Aumenta a fonte do campo de texto (chat input) */
        .stChatInput textarea {
            font-size: 16px !important;
        }
    </style>
""", unsafe_allow_html=True)

# Função de busca inteligente e flexível para planilhas
def buscar_materiais(df, termo_busca):
    if not termo_busca or df.empty:
        return pd.DataFrame()
    
    termos = termo_busca.lower().split()
    
    # Cria uma máscara para buscar os termos em qualquer coluna do DataFrame
    mascara = pd.Series(False, index=df.index)
    for col in df.columns:
        col_str = df[col].astype(str).str.lower()
        sub_mascara = pd.Series(True, index=df.index)
        for termo in termos:
            sub_mascara &= col_str.str.contains(termo, na=False, regex=False)
        mascara |= sub_mascara
        
    return df[mascara]

# Inicialização dos componentes de IA (com cache)
@st.cache_resource
def carregar_componentes_ia():
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    chroma_client = chromadb.PersistentClient(path="./db_docs")
    return encoder, chroma_client

encoder, chroma_client = carregar_componentes_ia()

# Configuração da API do OpenRouter
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
        st.info("### 🚂 Locomotivas")
        st.markdown("Suporte especializado em locomotivas, sistemas de freio CCBII e elétrica ferroviária.")
        if st.button("Acessar Locomotivas", type="primary"):
            st.session_state.sistema_ativo = "locomotivas"
            st.rerun()
            
        st.info("### 🛤️ Máquinas de Via")
        st.markdown("Suporte especializado em máquinas de via permanente, socadoras (Plasser) e hidráulica.")
        if st.button("Acessar Máquinas de Via", type="primary"):
            st.session_state.sistema_ativo = "maquinas_via"
            st.rerun()        
            
    with col2:
        st.info("### 🔍 Catálogo de Peças")
        st.markdown("Buscador inteligente de peças e part numbers em catálogos de fornecedores.")
        if st.button("Acessar Catálogo de Peças", type="primary"):
            st.session_state.sistema_ativo = "catalogo_pecas"
            st.rerun()
            
        st.info("### 📋 Código de Materiais")
        st.markdown("Consulta em planilha com códigos internos da empresa para requisição de materiais.")
        if st.button("Acessar Código de Materiais", type="primary"):
            st.session_state.sistema_ativo = "codigo_materiais"
            st.rerun()

# --- MÓDULO 3: BUSCADOR DE PEÇAS EM CATÁLOGOS (PDF) ---
elif st.session_state.sistema_ativo == "catalogo_pecas":
    if st.button("🏠 Voltar ao Menu Principal", type="secondary"):
        st.session_state.sistema_ativo = None
        st.rerun()
        
    st.title("🔍 Buscador de Peças em Catálogos (PDF)")
    
    pasta_catalogo = Path("./Docs_Catalogos")
    pasta_catalogo.mkdir(exist_ok=True)
    
    aba_cat = st.radio("Navegação Catálogo", ["🔍 Realizar Busca", "📂 Gerenciar Base de PDFs"], horizontal=True)
    
    if aba_cat == "🔍 Realizar Busca":
        termo_busca = st.text_input("Digite o nome da peça, descrição ou Part Number:")
        
        if st.button("Buscar no Catálogo", type="primary"):
            if not termo_busca:
                st.warning("Por favor, digite um termo para buscar.")
            else:
                with st.spinner("Consultando catálogos e analisando com IA..."):
                    arquivos_pdf = list(pasta_catalogo.glob("**/*.pdf"))
                    if not arquivos_pdf:
                        st.warning(f"Nenhum arquivo PDF encontrado na pasta `{pasta_catalogo}`. Vá na aba 'Gerenciar Base de PDFs' para adicionar.")
                    else:
                        contexto_partes = []
                        for pdf_path in arquivos_pdf:
                            nome_modelo_maquina = pdf_path.stem.replace("_", " ").replace("-", " ")
                            reader = pypdf.PdfReader(str(pdf_path))
                            for i, page in enumerate(reader.pages):
                                texto = page.extract_text()
                                if texto and termo_busca.lower() in texto.lower():
                                    contexto_partes.append(f"--- Modelo/Máquina (Arquivo): {nome_modelo_maquina} | Página: {i+1} ---\n{texto[:1200]}")
                        
                        if not contexto_partes:
                            st.info("Nenhum trecho direto correspondente foi encontrado nos arquivos. Tente um termo mais genérico.")
                        else:
                            contexto = "\n\n".join(contexto_partes[:10])
                            
                            prompt_sistema = """
                            Você é um agente especialista em suprimentos, engenharia e peças industriais ferroviárias.
                            Sua tarefa é analisar os trechos de catálogos fornecidos, identificar **todos** os itens correspondentes à busca do usuário e agrupar separadamente por máquina/modelo (cujo nome consta no início de cada trecho do contexto).
                            
                            Para CADA item encontrado, retorne em um formato limpo, estruturado e fácil de ler (em bullet points ou cards por máquina):
                            - **🚆 Modelo da Máquina / Equipamento:** [Nome extraído do arquivo]
                            - **🔩 Peça / Material:** [Nome claro do item]
                            - **🔢 Part Number / Código:** [Código encontrado no catálogo]
                            - **📄 Página de Referência:** [Número da página]
                            - **📝 Descrição / Detalhes:** [Breve resumo técnico]
                            
                            Exiba todos os resultados encontrados divididos por máquina. Seja organizado e objetivo.
                            """
                            
                            response = client.chat.completions.create(
                                model="openai/gpt-4o-mini",
                                messages=[
                                    {"role": "system", "content": prompt_sistema},
                                    {"role": "user", "content": f"Busca do usuário: {termo_busca}\n\nTrechos dos Catálogos:\n{contexto}"}
                                ],
                                temperature=0.0
                            )
                            st.markdown("### 📋 Resultados Encontrados por Máquina:")
                            st.markdown(response.choices[0].message.content)
    
    else:
        st.markdown("### 📂 Gerenciamento do Banco de Dados de Catálogos")
        st.markdown("Faça o upload de novos catálogos em PDF para atualizar a base de fornecedores. **Dica:** Nomeie o arquivo PDF com o modelo da máquina (ex: `Socadora_Plasser_08_475.pdf`), pois esse nome será usado para identificar a origem da peça.")
        
        uploaded_pdfs = st.file_uploader("Carregar novos catálogos (PDF)", type=["pdf"], accept_multiple_files=True, key="upload_pdf_cat")
        
        if uploaded_pdfs:
            for up_file in uploaded_pdfs:
                caminho_salvamento = pasta_catalogo / up_file.name
                with open(caminho_salvamento, "wb") as f:
                    f.write(up_file.getbuffer())
            st.success(f"{len(uploaded_pdfs)} catálogo(s) salvo(s) com sucesso!")
            
        st.divider()
        st.markdown("#### PDFs atualmente na base:")
        arquivos_atuais = list(pasta_catalogo.glob("**/*.pdf"))
        if arquivos_atuais:
            for arq in arquivos_atuais:
                col_p1, col_p2 = st.columns([0.8, 0.2])
                col_p1.text(f"📄 {arq.name}")
                if col_p2.button("Excluir", key=f"del_cat_{arq.name}"):
                    arq.unlink()
                    st.success(f"Arquivo {arq.name} removido!")
                    st.rerun()
        else:
            st.info("Nenhum PDF cadastrado no momento.")

# --- MÓDULO 4: CÓDIGO DE MATERIAIS (PLANILHA INTERNA) ---
elif st.session_state.sistema_ativo == "codigo_materiais":
    if st.button("🏠 Voltar ao Menu Principal", type="secondary"):
        st.session_state.sistema_ativo = None
        st.rerun()
        
    st.title("📋 Buscador de Código de Materiais (Planilha Interna)")
    
    pasta_excel = Path("./Docs_Planilhas")
    pasta_excel.mkdir(exist_ok=True)
    caminho_excel = pasta_excel / "materiais_internos.xlsx"
    
    aba_mat = st.radio("Navegação Materiais", ["📋 Realizar Consulta", "📂 Gerenciar Planilha de Dados"], horizontal=True)
    
    if aba_mat == "📋 Realizar Consulta":
        termo_interno = st.text_input("Digite o nome ou código interno do material (ex: Sensor DSS):")
        
        if st.button("Consultar Materiais", type="primary"):
            if not termo_interno:
                st.warning("Informe um termo para a consulta.")
            else:
                with st.spinner("Buscando na planilha interna..."):
                    if not caminho_excel.exists():
                        st.error(f"Nenhuma planilha encontrada em `{caminho_excel}`. Vá na aba 'Gerenciar Planilha de Dados' para fazer o upload.")
                    else:
                        df = pd.read_excel(caminho_excel)
                        df.columns = df.columns.str.strip()
                        
                        resultado = buscar_materiais(df, termo_interno)
                        
                        if not resultado.empty:
                            st.success(f"Foram encontrados {len(resultado)} item(ns):")
                            st.dataframe(resultado)
                        else:
                            st.warning(f"Nenhum item correspondente a '{termo_interno}' foi encontrado na planilha.")
    else:
        st.markdown("### 📂 Gerenciamento da Planilha de Materiais")
        st.markdown("Faça o upload da planilha atualizada contendo os códigos internos (`.xlsx` ou `.xls`).")
        
        uploaded_excel = st.file_uploader("Carregar planilha de materiais", type=["xlsx", "xls"], key="upload_excel_mat")
        
        if uploaded_excel:
            with open(caminho_excel, "wb") as f:
                f.write(uploaded_excel.getbuffer())
            st.success("Planilha de materiais atualizada com sucesso!")
            
        st.divider()
        if caminho_excel.exists():
            st.success(f"✅ Planilha ativa no sistema: `{caminho_excel.name}`")
            try:
                df_preview = pd.read_excel(caminho_excel)
                st.markdown("#### Pré-visualização dos dados:")
                st.dataframe(df_preview.head(10))
            except Exception as e:
                st.error(f"Erro ao ler a planilha: {e}")
        else:
            st.warning("⚠️ Nenhuma planilha de materiais carregada ainda.")

# --- MÓDULOS DE SUPORTE TÉCNICO (Locomotivas e Máquinas de Via) ---
else:
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
        st.session_state[chave_chats_falhas] = [{"id": 1, "titulo": "Nova Falha", "mensagens": [{"role": "assistant", "content": msg_inicial_falha}]}]
    if chave_chat_atual_falhas not in st.session_state:
        st.session_state[chave_chat_atual_falhas] = 0

    with st.sidebar:
        if st.button("🏠 Voltar ao Menu Principal", type="secondary"):
            st.session_state.sistema_ativo = None
            st.rerun()

        st.divider()
        st.markdown(f"## {titulo_sidebar}")
        st.caption("Inteligência em Manutenção")
        st.divider()

        aba_selecionada = st.radio(
            "Navegação",
            ["📖 Dúvidas Técnicas", "⚙️ Análise de Falhas", "📂 Adicionar Conhecimento"],
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

        elif aba_selecionada == "⚙️ Análise de Falhas":
            st.markdown("### 🕒 Histórico de Falhas")
            if st.button("➕ Novo Chat de Falhas"):
                lista_chats = st.session_state[chave_chats_falhas]
                novo_id = len(lista_chats) + 1
                lista_chats.append({
                    "id": novo_id, 
                    "titulo": f"Falha {novo_id}", 
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

        pergunta = st.chat_input("Qual dúvida técnica você tem hoje?", key="input_duvida")

        if pergunta:
            chat_atual["mensagens"].append({"role": "user", "content": pergunta})
            
            if chat_atual["titulo"].startswith("Nova Dúvida") or chat_atual["titulo"].startswith("Dúvida"):
                chat_atual["titulo"] = pergunta[:25] + "..." if len(pergunta) > 25 else pergunta

            with st.chat_message("user", avatar="👤"):
                st.markdown(pergunta)

            with st.chat_message("assistant", avatar="🔹"):
                with st.spinner("Realizando busca profunda nos manuais..."):
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
                            "explicando conceitos e detalhando especificações com base estrita nos manuais e documentos indexados na base de dados.\n\n"
                            "DIRETRIZES DE ATUAÇÃO:\n"
                            "1. Fidelidade ao Contexto: Utilize os trechos dos manuais fornecidos abaixo como sua fonte primária de verdade técnica.\n"
                            "2. Clareza e Estrutura: Explique os conceitos de forma didática, organizada em tópicos (bullet points) ou passos.\n"
                            "3. Transparência em Caso de Omissão: Se a resposta exata não constar, informe educadamente e dê uma orientação geral ressalvando que não consta no manual.\n"
                            "4. Citação de Fontes: Sempre cite o nome do documento (PDF) correspondente ao lado da informação.\n\n"
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

    elif aba_selecionada == "⚙️ Análise de Falhas":
        st.markdown(f"### ⚙️ Análise de Falhas - {st.session_state.sistema_ativo.capitalize()}")
        
        chat_idx = st.session_state[chave_chat_atual_falhas]
        chat_atual = st.session_state[chave_chats_falhas][chat_idx]

        for msg in chat_atual["mensagens"]:
            with st.chat_message("assistant" if msg["role"] == "assistant" else "user", avatar="🔹" if msg["role"] == "assistant" else "👤"):
                st.markdown(msg["content"])

        pergunta = st.chat_input("Descreva o equipamento, sistema e falha observada...", key="input_falha")

        if pergunta:
            chat_atual["mensagens"].append({"role": "user", "content": pergunta})
            
            if chat_atual["titulo"].startswith("Nova Falha") or chat_atual["titulo"].startswith("Falha"):
                chat_atual["titulo"] = pergunta[:25] + "..." if len(pergunta) > 25 else pergunta

            with st.chat_message("user", avatar="👤"):
                st.markdown(pergunta)

            with st.chat_message("assistant", avatar="🔹"):
                with st.spinner("Analisando falha, parâmetros e manuais..."):
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
                        
                        contexto = "\n\n".join(contexto_partes) if contexto_partes else "Nenhum trecho correspondente encontrado na base de PDFs local."
                        fontes_str = ", ".join(fontes_encontradas) if fontes_encontradas else "Nenhum manual PDF local indexado para citação."

                        system_prompt = (
                            "Você é o Mir, um técnico especialista sênior em manutenção ferroviária.\n"
                            "Sua linguagem é técnica, direta, prática e corporativa de oficina (voltada para técnicos e mecânicos).\n\n"
                            "DIRETRIZES DE RESPOSTA:\n"
                            "1. IDENTIFICAÇÃO DO MODELO: Preste atenção se o usuário citou o modelo do equipamento na pergunta.\n"
                            "2. ANÁLISE DE DADOS: Se o usuário descreveu parâmetros, analise detalhadamente.\n"
                            "3. HIPÓTESES PRIORIZADAS: Liste as prováveis causas em ordem decrescente de probabilidade.\n"
                            "4. AÇÃO PRÁTICA: O que o técnico deve fazer AGORA?\n\n"
                            "ESTRUTURA OBRIGATÓRIA DA RESPOSTA:\n"
                            "### 🔎 Interpretação do Evento\n"
                            "### 📊 Análise das Variáveis\n"
                            "### 💡 Minha Hipótese de Diagnóstico\n"
                            "### 🛠️ Plano de Ação (Checklist de Oficina)\n\n"
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
        st.markdown("Carregue novos manuais em PDF para expandir imediatamente a base de conhecimento.")

        uploaded_files = st.file_uploader(
            "Carregar manuais técnicos (PDF)", type=["pdf"], accept_multiple_files=True
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
            if st.button("🧠 Indexar Novos PDFs da Pasta", type="primary"):
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