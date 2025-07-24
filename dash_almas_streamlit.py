import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import base64
import os

from datetime import datetime, timedelta
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
from supabase import create_client, Client
from streamlit_autorefresh import st_autorefresh

# Atualiza a cada 15 minutos
st_autorefresh(interval=500 * 1000, key="auto_refresh")

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
def ler_dados_supabase(tabela: str, pagina_tamanho: int = 1000) -> pd.DataFrame:
    offset = 0
    dados_completos = []

    while True:
        resposta = (
            supabase
            .table(tabela)
            .select("*")
            .range(offset, offset + pagina_tamanho - 1)  # define o intervalo de linhas
            .execute()
        )
        dados = resposta.data
        if not dados:
            break  # terminou de puxar todas as linhas
        dados_completos.extend(dados)
        offset += pagina_tamanho

    return pd.DataFrame(dados_completos)

# Lê dados da tabela 'movimentacao_mina'
df_transporte_filtrado = ler_dados_supabase("movimentacao_mina")
df_totalizador = ler_dados_supabase("alimentacao_moagem")
df_vazao_final = ler_dados_supabase("alimentacao_planta_media_movel")
df_dados_planta = ler_dados_supabase("dados_planta")

# ==============================================
# Funções de agregação
# ==============================================

def agregar_por_hora(
    df,
    valor_coluna,
    coluna_hora='hora_completa',
    grupo_material=None,
    tipo_agregacao='sum'
):
    agora = datetime.now().replace(minute=0, second=0, microsecond=0)
    inicio = agora - timedelta(hours=28)

    df_filtrado = df.copy()

    # Garante que a coluna passada como 'coluna_hora' está no formato datetime e arredondada para hora cheia
    df_filtrado[coluna_hora] = pd.to_datetime(df_filtrado[coluna_hora], errors='coerce').dt.floor('h')

    # Filtra o intervalo de tempo: últimas 24h, excluindo a hora atual
    df_filtrado = df_filtrado[
        (df_filtrado[coluna_hora] >= inicio) & 
        (df_filtrado[coluna_hora] < agora)
    ]

    # Aplica o filtro de grupo de material, se fornecido
    if grupo_material is not None:
        df_filtrado = df_filtrado[df_filtrado['material_group'] == grupo_material]

    # Realiza a agregação por hora usando a coluna especificada
    df_agrupado = (
        df_filtrado
        .groupby(coluna_hora)[valor_coluna]
        .agg(tipo_agregacao)
        .reset_index()
        .rename(columns={coluna_hora: 'hora', valor_coluna: 'valor'})
    )

    return df_agrupado

# Função apra agregar dados a serem usados com a função de grafico empilhado
def agregar_por_hora_empilhado(
    df,
    valor_coluna,
    coluna_hora='hora_completa',
    coluna_empilhamento='material',
    tipo_agregacao='sum'
):
    agora = datetime.now().replace(minute=0, second=0, microsecond=0)
    inicio = agora - timedelta(hours=28)

    df_filtrado = df.copy()

    # Garante que a coluna de hora está formatada corretamente
    df_filtrado[coluna_hora] = pd.to_datetime(df_filtrado[coluna_hora], errors='coerce').dt.floor('h')

    # Filtro de tempo
    df_filtrado = df_filtrado[
        (df_filtrado[coluna_hora] >= inicio) & 
        (df_filtrado[coluna_hora] < agora)
    ]

    # Agregação por hora e coluna de empilhamento (ex: material)
    df_agrupado = (
        df_filtrado
        .groupby([coluna_hora, coluna_empilhamento])[valor_coluna]
        .agg(tipo_agregacao)
        .reset_index()
        .rename(columns={coluna_hora: 'hora', coluna_empilhamento: 'categoria', valor_coluna: 'valor'})
    )

    return df_agrupado

# ==============================================
# Funções para gerar graficos
# ==============================================

# Função para Criar os graficos de barras
def gerar_grafico_colunas(
    df_agrupado,
    valor_referencia=None,
    titulo='Título do Gráfico',
    yaxis_min=None,
    yaxis_max=None,
    tooltip_template=None
):
    df_plot = df_agrupado.copy()
    df_plot['hora_str'] = df_plot['hora'].dt.strftime('%H')
    df_plot['data'] = df_plot['hora'].dt.strftime('%d/%m')

    # Tooltip padrão atualizado
    if tooltip_template is None:
        tooltip_template = (
            "<b>Data</b>: %{customdata[0]}<br>"
            "<b>Hora</b>: %{x}<br>"
            "<b>Valor</b>: %{y:,}<extra></extra>"
        )

    # Aplica cores dependendo se há ou não valor_referencia
    if valor_referencia is not None:
        df_plot['cor'] = df_plot['valor'].apply(lambda x: "#F4614D" if x < valor_referencia else "#2D3D70")
    else:
        df_plot['cor'] = "#2D3D70"

    # Identifica índice de troca de dia
    dia_hoje = datetime.now().date()
    troca_idx = df_plot[df_plot['hora'].dt.date == dia_hoje].index.min()

    fig = go.Figure()

    # Barras com customdata para tooltip
    fig.add_trace(go.Bar(
        x=df_plot['hora_str'],
        y=df_plot['valor'],
        marker_color=df_plot['cor'],
        text=[f"{v:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".") for v in df_plot['valor']],
        textposition="inside",
        textangle=270,
        textfont=dict(color='white', size=25),
        hovertemplate=tooltip_template,
        customdata=df_plot[['data']].values,
    ))

    # Linha da meta (opcional)
    if valor_referencia is not None:
        fig.add_hline(
            y=valor_referencia,
            line_dash="dash",
            line_color="black",
            annotation_text=f"Meta: {valor_referencia:,.0f}".replace(",", "."),
            annotation_position="top right",
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
            y=1.09,
            xref='x',
            yref='paper',
            text=(dia_hoje - timedelta(days=1)).strftime('%d/%m'),
            showarrow=False,
            yanchor="top",
            font=dict(size=14, color="black")
        )
        fig.add_annotation(
            x=troca_idx + 0.5,
            y=1.09,
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
            tickangle=0,
            type='category',
            tickfont=dict(size=16, family='Arial', color='black'),
            showline=True,
            linecolor='black'
        ),
        yaxis=dict(
            visible=False,
            range=[yaxis_min, yaxis_max] if yaxis_min is not None and yaxis_max is not None else None
        ),
        bargap=0.2,
        margin=dict(t=40, b=20, l=0, r=0),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=300
    )

    return fig

# Função para criar grafico de barras empilhadas
def gerar_grafico_empilhado(
    df_agrupado,
    titulo='Título do Gráfico',
    legenda_yshift=20,
    yaxis_min=None,
    yaxis_max=None,
    cores_categorias=None,
    tooltip_template=None
):
    df_plot = df_agrupado.copy()

    # Ordena cronologicamente
    df_plot = df_plot.sort_values(['hora', 'categoria'])

    # Cria a string de hora (somente HH) para exibição no eixo x
    df_plot['hora_str'] = df_plot['hora'].dt.strftime('%H')

    # Garante ordenação com base em data/hora real
    categorias_x = df_plot.drop_duplicates('hora').sort_values('hora')['hora_str'].tolist()
    df_plot['hora_str'] = pd.Categorical(df_plot['hora_str'], categories=categorias_x, ordered=True)

    # Detecta a hora de troca de dia
    dia_hoje = datetime.now().date()
    troca_hora = df_plot[df_plot['hora'].dt.date == dia_hoje]['hora'].min()
    troca_hora_str = troca_hora.strftime('%H') if troca_hora is not None else None

    # Cores padrão
    if cores_categorias is None:
        cores_categorias = {
            'Estéril': '#AAAAAA',
            'HG': '#FF5733',
            'MG': '#FFC300',
            'LG': '#4CAF50',
            'HL': '#2D3D70'
        }

    categorias = list(cores_categorias.keys())

    if tooltip_template is None:
        tooltip_template = "<b>Hora</b>: %{x}<br><b>Valor</b>: %{y:,}<br><b>Categoria</b>: %{customdata[0]}<extra></extra>"

    fig = go.Figure()

    for cat in categorias:
        df_cat = df_plot[df_plot['categoria'] == cat]

        fig.add_trace(go.Bar(
            x=df_cat['hora_str'],
            y=df_cat['valor'],
            name=str(cat),
            marker_color=cores_categorias[cat],
            customdata=df_cat[['categoria']],
            hovertemplate=tooltip_template,
            showlegend=True
        ))

    # Rótulos de totais por hora_str
    df_totais = df_plot.groupby('hora_str', observed=True)['valor'].sum().reset_index()
    #df_totais['texto'] = df_totais['valor'].apply(lambda v: f"{v:,.0f}".replace(",", "X").replace(".", ",").replace("X", "."))
    df_totais['texto'] = df_totais['valor'].apply(lambda v: f"{v/1000:,.1f}".replace(",", "X").replace(".", ",").replace("X", "."))


    fig.add_trace(go.Scatter(
        x=df_totais['hora_str'],
        y=df_totais['valor'] + (df_totais['valor'].max() * 0.03),  # deslocamento para cima
        mode='text',
        text=df_totais['texto'],
        textposition='top center',
        showlegend=False,
        textfont=dict(size=14, color='black')
    ))

    # Linha de troca de dia (se aplicável)
    if troca_hora_str in categorias_x:
        troca_pos = categorias_x.index(troca_hora_str)

        fig.add_vline(
            x=troca_pos - 0.5,
            line_dash="solid",
            line_color="black",
            line_width=2
        )

        fig.add_annotation(
            x=troca_pos - 1.5,
            y=1.09,
            xref='x',
            yref='paper',
            text=(dia_hoje - timedelta(days=1)).strftime('%d/%m'),
            showarrow=False,
            yanchor="top",
            font=dict(size=14, color="black")
        )
        fig.add_annotation(
            x=troca_pos + 0.5,
            y=1.09,
            xref='x',
            yref='paper',
            text=dia_hoje.strftime('%d/%m'),
            showarrow=False,
            yanchor="top",
            font=dict(size=14, color="black")
        )

    fig.update_layout(
        barmode='stack',
        title=dict(
            text=titulo,
            x=0.0,
            xanchor='left',
            font=dict(size=20, family='Arial', color='black')
        ),
        xaxis=dict(
            tickangle=0,
            type='category',
            categoryorder='array',
            categoryarray=categorias_x,
            tickfont=dict(size=16, family='Arial', color='black'),
            showline=True,
            linecolor='black'
        ),
        yaxis=dict(
            visible=False,
            range=[yaxis_min, yaxis_max] if yaxis_min is not None and yaxis_max is not None else None
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.9 + (legenda_yshift / 100),
            xanchor="center",
            x=0.8,
            font=dict(size=14)
        ),
        bargap=0.2,
        margin=dict(t=40, b=20, l=0, r=0),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=300
    )

    return fig

# =========================================
# Criação dos Graficos MINA - Movimentação
# =========================================

# Selecionar def com o conteudo
df_base_mina = df_transporte_filtrado

# Grafico 1 - Contagem de Viajens
df_agg_viagens = agregar_por_hora(
    df=df_base_mina,
    valor_coluna='calculated_mass',
    grupo_material=None,
    tipo_agregacao= 'count'
)

grafico_numero_viagens = gerar_grafico_colunas(
    df_agrupado=df_agg_viagens,
    valor_referencia=71,
    titulo='Numero de Viagens',
    yaxis_min=0
)

# Grafico 2 - Movimentação Total por litologia
# Define cores e ordem desejada
cores_customizadas = {
    'HG': '#FF5733',
    'MG': '#FFC300',
    'LG': '#4CAF50',
    'HL': '#2D3D70',
    'Estéril': '#AAAAAA'
}

# Agrega os dados
df_agg_movimentacao_litologia = agregar_por_hora_empilhado(
    df=df_base_mina,
    valor_coluna='calculated_mass',
    coluna_empilhamento='material',
    tipo_agregacao='sum'
)

# Gera gráfico com tooltip customizado
grafico_movimentacao_litogia = gerar_grafico_empilhado(
    df_agrupado=df_agg_movimentacao_litologia,
    titulo='Movimentação por litologia (Kt)',
    legenda_yshift=20,
    yaxis_min=0,
    cores_categorias=cores_customizadas,
    #tooltip_template="<b>Material</b>: %{customdata[0]}<br><b>Hora</b>: %{x}h<br><b>Valor</b>: %{y:,} toneladas<extra></extra>"
    tooltip_template = "<b>Material</b>: %{customdata[0]}<br><b>Hora</b>: %{x}h<br><b>Valor</b>: %{y:,.2f} toneladas<extra></extra>"

)

# =====================================
# Criação dos Graficos Planta 
# =====================================

# Selecionar def com o conteudo
df_base_planta = df_dados_planta

# Grafico 1 - Alimentação Britagem
df_agg_britagem = agregar_por_hora(
    df=df_base_planta,
    coluna_hora='Timestamp',
    valor_coluna='Britagem_Massa Produzida Britagem_(t)',
    tipo_agregacao='sum'
)

grafico_barra_britagem = gerar_grafico_colunas(
    df_agrupado=df_agg_britagem,
    valor_referencia=310,  
    titulo='Alimentação Britagem (t)',
    yaxis_min=0,
    yaxis_max=420
)

# Grafico 2 - Alimentação Moagem
df_agg_moagem = agregar_por_hora(
    df=df_base_planta,
    coluna_hora='Timestamp',
    valor_coluna='Moinho_Massa Alimentada Moagem_(t)',
    tipo_agregacao='sum'
)

grafico_barra_moagem = gerar_grafico_colunas(
    df_agrupado=df_agg_moagem,
    valor_referencia=250,  
    titulo='Alimentação Moagem (t)',
    yaxis_min=0,
    yaxis_max=420
)

# =============================================
# Dashboard em Streamlit - Desenvolvimento
# =============================================

# ========== Configuração da página ==========
st.set_page_config(layout="wide")

# ========== Estilo CSS otimizado para Full HD ==========
st.markdown("""
    <style>
        .block-container {
            padding-top: 0rem !important;
            padding-bottom: 0rem !important;
            max-width: 1900px;
            margin: auto;
        }
        header, .main {
            padding-top: 0.5rem !important;
        }
        h1, h2, h3 {
            margin-top: -10px !important;
            margin-bottom: 0px !important;
            padding: 0 !important;
        }
        .stPlotlyChart {
            padding: 0 !important;
            margin: 0 !important;
        }
    </style>
""", unsafe_allow_html=True)

# ========== Carregamento de logos ==========
pasta_atual = os.path.dirname(__file__)
logo_aura = os.path.join(pasta_atual, "Icones", "Logo_Aura.jpg")
logo_mina = os.path.join(pasta_atual, "Icones", "caminhao.png")
logo_moagem = os.path.join(pasta_atual, "Icones", "mill.png")

def imagem_para_base64_e_tipo(caminho_imagem):
    imagem = Image.open(caminho_imagem)
    buffer = BytesIO()
    extensao = os.path.splitext(caminho_imagem)[1].lower()
    if extensao in ['.jpg', '.jpeg']:
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

base64_esquerda, tipo_esquerda = imagem_para_base64_e_tipo(logo_mina)
base64_esquerda2, tipo_esquerda2 = imagem_para_base64_e_tipo(logo_moagem)
base64_direita, tipo_direita = imagem_para_base64_e_tipo(logo_aura)

# =============================================
# Renderização do Dashboard em Tela Única (Full HD)
# =============================================

# Cabeçalho Mina
st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center;
        background-color: #2D3D70; padding: 0px 30px; border-radius: 8px; margin-top: -10px; margin-bottom: 5px;">
        <img src="data:{tipo_esquerda};base64,{base64_esquerda}" style="height: 45px;">
        <h1 style="color: white; font-size: 28px; margin: 0;">Performance Mina Paiol - Aura Almas</h1>
        <img src="data:{tipo_direita};base64,{base64_direita}" style="height: 40px;">
    </div>
""", unsafe_allow_html=True)

# Linha 1 - Total / Minério
col1, col2 = st.columns([0.5, 0.5], gap="large")
with col1:
    if not df_agg_viagens.empty:
        st.plotly_chart(grafico_numero_viagens.update_layout(height=300), use_container_width=True)
with col2:
    if not df_agg_viagens.empty:
        st.plotly_chart(grafico_movimentacao_litogia.update_layout(height=300), use_container_width=True)

# Linha 2 - Viagens / Estéril
col3, col4 = st.columns([0.5, 0.5], gap="large")

# Cabeçalho Moagem
st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center;
        background-color: #2D3D70; padding: 0px 30px; border-radius: 8px; margin-top: 100px; margin-bottom: 5px;">
        <img src="data:{tipo_esquerda2};base64,{base64_esquerda2}" style="height: 40px;">
        <h1 style="color: white; font-size: 28px; margin: 0;">Performance Planta - Aura Almas</h1>
        <img src="data:{tipo_direita};base64,{base64_direita}" style="height: 40px;">
    </div>
""", unsafe_allow_html=True)

# Linha 3 - Totalizador / Média Móvel
col5, col6 = st.columns([0.5, 0.5], gap="large")
with col5:
    if not df_agg_britagem.empty:
        st.plotly_chart(grafico_barra_britagem.update_layout(height=300), use_container_width=True)
with col6:
    if not df_agg_moagem.empty:
        st.plotly_chart(grafico_barra_moagem.update_layout(height=300), use_container_width=True)