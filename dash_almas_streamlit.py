import requests
import pandas as pd
import urllib3
import plotly.graph_objects as go
import streamlit as st
import base64
import os
import time

from sqlalchemy import create_engine
from datetime import datetime, timedelta
from PIL import Image
from streamlit_autorefresh import st_autorefresh
from io import BytesIO
from dotenv import load_dotenv
from supabase import create_client, Client

# ==============================================
# Carregamento das tabelas
# ==============================================

# Carrega as variáveis do arquivo .env
load_dotenv()

# Agora pega as variáveis pelo nome
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Inicializa cliente
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Função para ler os dados das tabelas
def ler_dados_supabase(tabela: str) -> pd.DataFrame:
    """
    Lê todos os dados de uma tabela no Supabase e retorna um DataFrame.
    """
    resposta = supabase.table(tabela).select("*").execute()
    dados = resposta.data
    return pd.DataFrame(dados)

# Lê dados da tabela 'movimentacao_mina'
df_transporte_filtrado = ler_dados_supabase("movimentacao_mina")
df_totalizador = ler_dados_supabase("alimentacao_moagem")
df_vazao_final = ler_dados_supabase("alimentacao_planta_media_movel")

# ==============================================
# Funções para gerar dados e os graficos de mina
# ==============================================
# Função para agregar os dados
def agregar_por_hora_mina(
    df,
    valor_coluna,
    coluna_hora='hora_completa',
    grupo_material=None,
    tipo_agregacao='sum'
):
    agora = datetime.now().replace(minute=0, second=0, microsecond=0)
    inicio = agora - timedelta(hours=24)

    df_filtrado = df.copy()

    # Garante que a coluna de hora está arredondada corretamente
    df_filtrado[coluna_hora] = pd.to_datetime(df_filtrado[coluna_hora], errors='coerce').dt.floor('h')

    # Filtro de tempo - exclui hora atual para evitar duplicidade
    df_filtrado = df_filtrado[
        (df_filtrado[coluna_hora] >= inicio) & 
        (df_filtrado[coluna_hora] < agora)
    ]

    # Filtro por grupo de material
    if grupo_material:
        df_filtrado = df_filtrado[df_filtrado['material_group'] == grupo_material]

    # Agregação
    df_agrupado = (
        df_filtrado
        .groupby(coluna_hora)[valor_coluna]
        .agg(tipo_agregacao)
        .reset_index()
        .rename(columns={coluna_hora: 'hora', valor_coluna: 'valor'})
    )

    return df_agrupado

# Função para Criar os graficos mina
def gerar_grafico_colunas_mina(df_agrupado, valor_referencia=None, titulo='Título do Gráfico', yaxis_min=None, yaxis_max=None):
    df_plot = df_agrupado.copy()
    df_plot['hora_str'] = df_plot['hora'].dt.strftime('%H:%M')
    df_plot['data'] = df_plot['hora'].dt.strftime('%d/%m')

    # Aplica cores dependendo se há ou não valor_referencia
    if valor_referencia is not None:
        df_plot['cor'] = df_plot['valor'].apply(lambda x: "#F4614D" if x < valor_referencia else "#2D3D70")
    else:
        df_plot['cor'] = "#2D3D70"

    # Identifica índice de troca de dia
    dia_hoje = datetime.now().date()
    troca_idx = df_plot[df_plot['hora'].dt.date == dia_hoje].index.min()

    fig = go.Figure()

    # Barras
    fig.add_trace(go.Bar(
        x=df_plot['hora_str'],
        y=df_plot['valor'],
        marker_color=df_plot['cor'],
        text=[f"{v:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".") for v in df_plot['valor']],
        textposition="inside",
        textangle=270,
        textfont=dict(color='white'),
        hovertemplate="<b>Hora</b>: %{x}<br><b>Valor</b>: %{y:,}<extra></extra>",
    ))

    # Linha da meta (opcional)
    if valor_referencia is not None:
        fig.add_hline(
            y=valor_referencia,
            line_dash="dash",
            line_color="black",
            annotation_text=f"Meta: {valor_referencia:,.0f}".replace(",", "."),
            annotation_position="top left",
            annotation_font_size=12,
            annotation_font_color="black",
            annotation_yshift=30
        )

    # Linha vertical da troca de dia + rótulos de datas abaixo do eixo X
    if troca_idx is not None and troca_idx > 0:
        fig.add_vline(
            x=troca_idx - 0.5,
            line_dash="solid",
            line_color="black",
            line_width=2
        )
        fig.add_annotation(
            x=troca_idx - 1.5,
            y=-0.3,
            xref='x',
            yref='paper',
            text=(dia_hoje - timedelta(days=1)).strftime('%d/%m'),
            showarrow=False,
            yanchor="top",
            font=dict(size=14, color="black")
        )
        fig.add_annotation(
            x=troca_idx + 0.5,
            y=-0.3,
            xref='x',
            yref='paper',
            text=dia_hoje.strftime('%d/%m'),
            showarrow=False,
            yanchor="top",
            font=dict(size=14, color="black")
        )

    # Layout final
    fig.update_layout(
        title=dict(
            text=titulo,
            x=0.0,
            xanchor='left',
            font=dict(size=20, family='Arial', color='black')
        ),
        xaxis=dict(
            tickangle=-45,
            tickfont=dict(size=12, family='Arial', color='black'),
            showline=True,
            linecolor='black'
        ),
        yaxis=dict(
            visible=False,
            range=[yaxis_min, yaxis_max] if yaxis_min is not None and yaxis_max is not None else None
        ),
        bargap=0.2,
        margin=dict(t=40, b=100, l=0, r=0),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=300
    )
    return fig

# =============================
# Gerar Graficos dados de MINA
# =============================

# Selecionar def com o conteudo
df_base_mina = df_transporte_filtrado

# Grafico 1 - Movimentação Total
df_agg_total = agregar_por_hora_mina(
    df=df_base_mina,
    valor_coluna='calculated_mass',
    grupo_material=None,
    tipo_agregacao= 'sum'
)

grafico_total = gerar_grafico_colunas_mina(
    df_agrupado=df_agg_total,
    valor_referencia=(71 * 50),
    titulo='Movimentação Total',
    yaxis_min=0
)
# Grafico 2 - Contagem de Viajens
df_agg_viagens = agregar_por_hora_mina(
    df=df_base_mina,
    valor_coluna='calculated_mass',
    grupo_material=None,
    tipo_agregacao= 'count'
)

grafico_numero_viagens = gerar_grafico_colunas_mina(
    df_agrupado=df_agg_viagens,
    valor_referencia=71,
    titulo='Numero de Viagens',
    yaxis_min=0
)

# Grafico 3 - Movimentação de Minério
df_agg_minerio = agregar_por_hora_mina(
    df=df_base_mina,
    valor_coluna='calculated_mass',
    grupo_material='Minério',
    tipo_agregacao= 'sum'
)

grafico_minerio = gerar_grafico_colunas_mina(
    df_agrupado=df_agg_minerio,
    valor_referencia=None,
    titulo='Movimentação de Minério',
    yaxis_min=0
)

# Grafico 4 - Movimentação de Estéril
df_agg_esteril = agregar_por_hora_mina(
    df=df_base_mina,
    valor_coluna='calculated_mass',
    grupo_material='Estéril',
    tipo_agregacao= 'sum'
)

grafico_esteril = gerar_grafico_colunas_mina(
    df_agrupado=df_agg_esteril,
    valor_referencia=None,
    titulo='Movimentação de Esteril',
    yaxis_min=0
)

# ================================
# MODULO DADOS DA PLANTA
# ================================

#====================================================================
# Preparar dados para criar os graficos de barras com dados da planta
#====================================================================

# Função para processar Produção Hora a Hora da moagem
def agregar_por_hora_planta(
    df,
    valor_coluna="Retomada - TR02 - Balança",
    tipo_agregacao='sum'
):

    if df.empty:
        return pd.DataFrame(columns=['hora', 'valor'])

    df = df.copy()

    # Cria a coluna de hora completa com arredondamento para a hora cheia
    df['hora_completa'] = pd.to_datetime(df['Timestamp'], errors='coerce').dt.floor('h')
    
    # Filtra apenas as últimas 24h completas, excluindo a hora atual
    agora = datetime.now().replace(minute=0, second=0, microsecond=0)
    inicio = agora - timedelta(hours=24)

    df_filtrado = df[
        (df['hora_completa'] >= inicio) & 
        (df['hora_completa'] < agora)
    ].copy()

    # Converte a coluna de valor para numérica (forçando erro para NaN)
    df_filtrado[valor_coluna] = pd.to_numeric(df_filtrado[valor_coluna], errors='coerce')

    # Remove linhas com valores nulos
    df_filtrado = df_filtrado.dropna(subset=[valor_coluna, 'hora_completa'])

    # Agrega por hora
    df_agrupado = (
        df_filtrado
        .groupby('hora_completa')[valor_coluna]
        .agg(tipo_agregacao)
        .reset_index()
        .rename(columns={'hora_completa': 'hora', valor_coluna: 'valor'})
    )

    return df_agrupado

# Função para Criar os graficos com dados da planta
def gerar_grafico_colunas_planta(df_agrupado, valor_referencia=None, titulo='Título do Gráfico', yaxis_min=None, yaxis_max=None):

    df_plot = df_agrupado.copy()
    df_plot['hora_str'] = df_plot['hora'].dt.strftime('%H:%M')
    df_plot['data'] = df_plot['hora'].dt.strftime('%d/%m')

    # Aplica cores dependendo se há ou não valor_referencia
    if valor_referencia is not None:
        df_plot['cor'] = df_plot['valor'].apply(lambda x: "#F4614D" if x < valor_referencia else "#2D3D70")
    else:
        df_plot['cor'] = "#2D3D70"

    # Identifica índice de troca de dia
    dia_hoje = datetime.now().date()
    troca_idx = df_plot[df_plot['hora'].dt.date == dia_hoje].index.min()

    fig = go.Figure()

    # Barras
    fig.add_trace(go.Bar(
        x=df_plot['hora_str'],
        y=df_plot['valor'],
        marker_color=df_plot['cor'],
        text=[f"{v:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".") for v in df_plot['valor']],
        textposition="inside",
        textangle=270,
        textfont=dict(color='white'),
        hovertemplate="<b>Hora</b>: %{x}<br><b>Valor</b>: %{y:,}<extra></extra>",
    ))

    # Linha da meta (opcional)
    if valor_referencia is not None:
        fig.add_hline(
            y=valor_referencia,
            line_dash="dash",
            line_color="black",
            annotation_text=f"Meta: {valor_referencia:,.0f}".replace(",", "."),
            annotation_position="top left",
            annotation_font_size=12,
            annotation_font_color="black",
            annotation_yshift=20
        )

    # Linha vertical da troca de dia + rótulos de datas abaixo do eixo X
    if troca_idx is not None and troca_idx > 0:
        fig.add_vline(
            x=troca_idx - 0.5,
            line_dash="solid",
            line_color="black",
            line_width=2
        )

        fig.add_annotation(
            x=troca_idx - 1.5,
            y=-0.3,
            xref='x',
            yref='paper',
            text=(dia_hoje - timedelta(days=1)).strftime('%d/%m'),
            showarrow=False,
            yanchor="top",
            font=dict(size=14, color="black")
        )
        fig.add_annotation(
            x=troca_idx + 0.5,
            y=-0.3,
            xref='x',
            yref='paper',
            text=dia_hoje.strftime('%d/%m'),
            showarrow=False,
            yanchor="top",
            font=dict(size=14, color="black")
        )

    # Layout final
    fig.update_layout(
        title=dict(
            text=titulo,
            x=0.0,
            xanchor='left',
            font=dict(size=20, family='Arial', color='black')
        ),
        xaxis=dict(
            tickangle=-45,
            tickfont=dict(size=12, family='Arial', color='black'),
            showline=True,
            linecolor='black'
        ),
        yaxis=dict(
            visible=False,
            range=[yaxis_min, yaxis_max] if yaxis_min is not None and yaxis_max is not None else None
        ),
        bargap=0.2,
        margin=dict(t=40, b=100, l=0, r=0),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=300
    )

    return fig

# =============================================
# Gerar Graficos de Barras com dados da planta
# =============================================

# Selecionar def com o conteudo
df_base_planta = df_totalizador

# Grafico 1 - Alimentação Moagem
df_agg_totalizador = agregar_por_hora_planta(
    df=df_base_planta,
    valor_coluna='Retomada - TR02 - Balança',
    tipo_agregacao='sum'
)

grafico_totalizador = gerar_grafico_colunas_planta(
    df_agrupado=df_agg_totalizador,
    valor_referencia=250,  
    titulo='Alimentação Moagem',
    yaxis_min=0,
    yaxis_max=307
)

#=====================================================================
# Preparar dados para criar os graficos de barras com dados da planta
#=====================================================================
def preparar_dados_linha_planta(df, coluna_valor):

    if df.empty or coluna_valor not in df.columns:
        return pd.DataFrame(columns=['Timestamp', 'hora', 'valor'])

    df = df.copy()
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df.dropna(subset=['Timestamp', coluna_valor], inplace=True)
    df['Timestamp'] = df['Timestamp'].dt.tz_localize(None)

    agora = datetime.now().replace(minute=0, second=0, microsecond=0)
    inicio = agora - timedelta(hours=24)

    # Arredonda Timestamp para hora cheia para filtro
    df['hora_completa'] = df['Timestamp'].dt.floor('h')

    # Aplica filtro com intervalo [inicio, agora)
    df = df[(df['hora_completa'] >= inicio) & (df['hora_completa'] < agora)]

    # Mantém coluna hora arredondada para hora cheia para uso posterior
    df['hora'] = df['hora_completa']

    df = df.rename(columns={coluna_valor: 'valor'})

    return df[['Timestamp', 'hora', 'valor']]

# Função apara gerar grafico de linha com dados da planta
def gerar_grafico_linha_planta(df_linha, titulo='Título do Gráfico', yaxis_min=None, yaxis_max=None, valor_referencia=None):

    df_plot = df_linha.copy()
    df_plot = df_plot.sort_values(by='Timestamp')  # Garante ordem cronológica real

    fig = go.Figure()

    # Linha de tendência (dados)
    fig.add_trace(go.Scatter(
        x=df_plot['Timestamp'],
        y=df_plot['valor'],
        mode='lines',
        line=dict(color='#3C4788', width=2),
        hovertemplate="<b>Hora</b>: %{x|%d/%m %H:%M}<br><b>Valor</b>: %{y:.2f}<extra></extra>",
    ))

    # Linha da meta
    if valor_referencia is not None:
        fig.add_hline(
            y=valor_referencia,
            line_dash="dash",
            line_color="black",
            annotation_text=f"Meta: {valor_referencia:,.0f}".replace(",", "."),
            annotation_position="top left",
            annotation_font_size=12,
            annotation_font_color="black",
            annotation_yshift=30
        )

    # Linha vertical da troca de dia
    dia_hoje = datetime.now().date()
    troca_idx = df_plot[df_plot['Timestamp'].dt.date == dia_hoje].index.min()
    if troca_idx is not None and troca_idx > 0:
        fig.add_vline(
            x=df_plot.loc[troca_idx, 'Timestamp'],
            line_dash="solid",
            line_color="black",
            line_width=2
        )
        fig.add_annotation(
            x=df_plot.loc[troca_idx, 'Timestamp'] - timedelta(hours=1),
            y=-0.3,
            xref='x',
            yref='paper',
            text=(dia_hoje - timedelta(days=1)).strftime('%d/%m'),
            showarrow=False,
            yanchor="top",
            font=dict(size=14, color="black")
        )
        fig.add_annotation(
            x=df_plot.loc[troca_idx, 'Timestamp'] + timedelta(hours=1),
            y=-0.3,
            xref='x',
            yref='paper',
            text=dia_hoje.strftime('%d/%m'),
            showarrow=False,
            yanchor="top",
            font=dict(size=14, color="black")
        )

    # Layout com escala horária
    fig.update_layout(
        title=dict(
            text=titulo,
            x=0.0,
            xanchor='left',
            font=dict(size=20, family='Arial', color='black')
        ),
        xaxis=dict(
            type="date",
            tickformat="%H:%M",
            tickmode="linear",
            dtick=3600000,  # 1 hora em milissegundos
            tickangle=-45,
            tickfont=dict(size=12, family='Arial', color='black'),
            showline=True,
            linecolor='black'
        ),
        yaxis=dict(
            visible=False,
            range=[yaxis_min, yaxis_max] if yaxis_min is not None and yaxis_max is not None else None
        ),
        margin=dict(t=40, b=100, l=0, r=0),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=300
    )
    return fig

# =============================================
# Gerar Graficos de linhas com dados da planta
# =============================================
# Prepara os dados da linha com base no df_vazao_24h
df_linha_planta = preparar_dados_linha_planta(df_vazao_final,'Retomada - TR02 - Balança_mm200s')

# Gera o gráfico de linha com layout personalizado
grafico_linha = gerar_grafico_linha_planta(
    df_linha_planta,
    titulo="Média Móvel Alimentação (t/h)",
    valor_referencia=250,
    yaxis_min=0,
    yaxis_max=307
)
# =============================================
# Dashboard em Streamlit
# =============================================
# ========== Configuração da página ==========
st.set_page_config(layout="wide")

# ========== Reduzir espaçamento do topo da página ==========
# CSS para reduzir espaços verticais
st.markdown("""
    <style>
        .block-container {
            padding-top: 0rem !important;
            padding-bottom: 0rem !important;
        }
        .main {
            padding-top: 0rem !important;
        }
        header {
            margin: 0rem !important;
            padding: 0rem !important;
        }
        h1, h2, h3 {
            margin-top: 5px !important;
            margin-bottom: 5px !important;
            padding: 0 !important;
        }
        .element-container {
            margin-bottom: 0rem !important;
            padding-bottom: 0rem !important;
        }
        .stPlotlyChart {
            padding: 0rem !important;
            margin-top: 0rem !important;
            margin-bottom: 0rem !important;
        }
    </style>
""", unsafe_allow_html=True)

# Carrega a imagem do logo
pasta_atual = os.path.dirname(__file__)  # pasta onde está o script atual

logo_aura = os.path.join(pasta_atual, "Icones", "Logo_Aura.jpg")
logo_mina = os.path.join(pasta_atual, "Icones", "escavadora.png")
logo_moagem = os.path.join(pasta_atual, "Icones", "mill.png")

# Função para converter imagem para base64 e retornar também o tipo MIME correto
def imagem_para_base64_e_tipo(caminho_imagem):
    imagem = Image.open(caminho_imagem)
    buffer = BytesIO()
    
    extensao = os.path.splitext(caminho_imagem)[1].lower()
    if extensao == '.jpg' or extensao == '.jpeg':
        formato = 'JPEG'
        mime_type = 'image/jpeg'
    elif extensao == '.png':
        formato = 'PNG'
        mime_type = 'image/png'
    else:
        raise ValueError(f"Formato de imagem não suportado: {extensao}")
    
    imagem.save(buffer, format=formato)
    imagem_base64 = base64.b64encode(buffer.getvalue()).decode()
    return imagem_base64, mime_type

# Geração das imagens com tipo MIME
base64_esquerda, tipo_esquerda = imagem_para_base64_e_tipo(logo_mina)
base64_esquerda2, tipo_esquerda2 = imagem_para_base64_e_tipo(logo_moagem)
base64_direita, tipo_direita = imagem_para_base64_e_tipo(logo_aura)

# HTML do cabeçalho estilizado mina
st.markdown(f"""
    <div style="
        display: flex;
        justify-content: space-between;
        align-items: center;
        background-color: #2D3D70;
        padding: 0px 30px;
        border-radius: 8px;
        margin-bottom: 10px;
    ">
        <img src="data:{tipo_esquerda};base64,{base64_esquerda}" style="height: 50px;">
        <h1 style="color: white; font-size: 35px; margin: 0;">
            Movimentação Mina Paiol - Aura Almas
        </h1>
        <img src="data:{tipo_direita};base64,{base64_direita}" style="height: 70px;">
    </div>
""", unsafe_allow_html=True)

# Pequeno espaço negativo para colar os gráficos no título
st.markdown('<div style="margin-top: -50px;"></div>', unsafe_allow_html=True)

# Primeira linha de gráficos lado a lado (mina)
col1, col2 = st.columns([1, 1], gap="medium")

with col1:
    if not df_agg_total.empty:
        st.plotly_chart(grafico_total, use_container_width=False)

with col2:
    if not df_agg_minerio.empty:
        st.plotly_chart(grafico_minerio, use_container_width=False)

# Reduz espaço entre primeira e segunda linha de colunas
st.markdown('<div style="margin-top: -50px;"></div>', unsafe_allow_html=True)

# Segunda linha de gráficos lado a lado (mina)
col3, col4 = st.columns([1, 1], gap="medium")

with col3:
    if not df_agg_viagens.empty:
        st.plotly_chart(grafico_numero_viagens, use_container_width=True)

with col4:
    if not df_agg_esteril.empty:
        st.plotly_chart(grafico_esteril, use_container_width=True)

# HTML do cabeçalho estilizado moagem
st.markdown(f"""
    <div style="
        display: flex;
        justify-content: space-between;
        align-items: center;
        background-color: #2D3D70;
        padding: 0px 30px;
        border-radius: 8px;
        margin-bottom: 10px;
    ">
        <img src="data:{tipo_esquerda2};base64,{base64_esquerda2}" style="height: 70px;">
        <h1 style="color: white; font-size: 35px; margin: 0;">
            Alimentação Moagem - Aura Almas
        </h1>
        <img src="data:{tipo_direita};base64,{base64_direita}" style="height: 70px;">
    </div>
""", unsafe_allow_html=True)

# Pequeno espaço negativo para colar os gráficos no título
st.markdown('<div style="margin-top: -100px;"></div>', unsafe_allow_html=True)

#Terceira linha de gráficos lado a lado (planta)
col5, col6 = st.columns([1, 1], gap="medium")

with col5:
    if not df_agg_totalizador.empty:
        st.plotly_chart(grafico_totalizador, use_container_width=True)

with col6:
    if not df_linha_planta.empty:
        st.plotly_chart(grafico_linha, use_container_width=True)