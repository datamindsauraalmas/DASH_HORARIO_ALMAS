import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import base64
import os
import pytz

from datetime import datetime, timedelta
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
from supabase import create_client, Client
from streamlit_autorefresh import st_autorefresh

# Atualiza a cada 5 minutos
st_autorefresh(interval=250 * 1000, key="auto_refresh")

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
df_dados_planta = ler_dados_supabase("dados_planta")
    
# Renomer nomes das colunas para melhor exibição no Tooltip dos graficos
df_dados_planta.rename(columns={
    "Moinho_Justificativa do Tempo operando com taxa a menor_(txt)": "Desvio taxa Moagem",
    "Britagem_Justificativa de NÂO atingir a massa_(txt)": "Justificativa Alimentação Britagem",
    "Moinho_Justificativa de NÂO atingir a massa_(txt)": "Justificativa Alimentação Moagem"
}, inplace=True)

# ==============================================
# Funções de agregação
# ==============================================

#Parametros para filtrar os dados dos graficos com base nas ultimas 24 horas
parametro_agora = datetime.now().replace(minute=0, second=0, microsecond=0)-timedelta(hours=3)
parametro_inicio = parametro_agora - timedelta(hours=24)

def agregar_por_hora(
    df,
    valor_coluna,
    coluna_hora='hora_completa',
    grupo_material=None,
    tipo_agregacao='sum',
    colunas_texto=None
):
    agora = parametro_agora
    inicio = parametro_inicio

    df_filtrado = df.copy()
    df_filtrado[coluna_hora] = pd.to_datetime(df_filtrado[coluna_hora], errors='coerce').dt.floor('h')
    df_filtrado = df_filtrado[(df_filtrado[coluna_hora] >= inicio) & (df_filtrado[coluna_hora] < agora)]

    if grupo_material is not None:
        df_filtrado = df_filtrado[df_filtrado['material_group'] == grupo_material]

    # Colunas para manter além da hora e valor
    colunas_agregadas = [coluna_hora, valor_coluna]
    if colunas_texto:
        colunas_agregadas += colunas_texto

    df_filtrado = df_filtrado[colunas_agregadas]

    # Realiza agregação
    df_agrupado = (
        df_filtrado
        .groupby(coluna_hora)
        .agg({valor_coluna: tipo_agregacao, **{col: 'first' for col in (colunas_texto or [])}})
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
    agora = parametro_agora
    inicio = parametro_inicio

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
    colunas_tooltip=None
):
    df_plot = df_agrupado.copy()
    df_plot['hora_str'] = df_plot['hora'].dt.strftime('%H')
    df_plot['data'] = df_plot['hora'].dt.strftime('%d/%m')

    # Geração do campo customdata com base nas colunas fornecidas
    colunas_tooltip = colunas_tooltip or []
    customdata = []

    for _, row in df_plot.iterrows():
        linha_tooltip = []
        for col in colunas_tooltip:
            valor = row.get(col)
            if pd.notnull(valor):
                linha_tooltip.append(f"<b>{col}</b>: {valor}")
        linha_tooltip.insert(0, f"<b>Data</b>: {row['data']}")  # sempre adiciona a data
        linha_tooltip.append(f"<b>Hora</b>: {row['hora_str']}")
        linha_tooltip.append(f"<b>Valor</b>: {row['valor']:,.0f}".replace(",", "X").replace(".", ",").replace("X", "."))
        customdata.append("<br>".join(linha_tooltip))

    # Aplica cores dependendo se há ou não valor_referencia
    if valor_referencia is not None:
        df_plot['cor'] = df_plot['valor'].apply(lambda x: "#F4614D" if x < valor_referencia else "#2D3D70")
    else:
        df_plot['cor'] = "#2D3D70"

    # Identifica índice de troca de dia
    dia_hoje = datetime.now().date()
    troca_idx = df_plot[df_plot['hora'].dt.date == dia_hoje].index.min()

    fig = go.Figure()

    # Barras com texto personalizado no hover
    fig.add_trace(go.Bar(
        x=df_plot['hora_str'],
        y=df_plot['valor'],
        marker_color=df_plot['cor'],
        text=[f"{v:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".") for v in df_plot['valor']],
        textposition="inside",
        textangle=270,
        textfont=dict(color='white', size=25),
        hovertemplate="%{customdata}<extra></extra>",
        customdata=customdata
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
            annotation_yshift=50
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

# Remove duplicatas e mantém ordem cronológica
    categorias_x = list(dict.fromkeys(df_plot.sort_values('hora')['hora_str'].tolist()))

# Converte para categórico com categorias ordenadas
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
            y=0.85 + (legenda_yshift / 100),
            xanchor="center",
            x=0.8,
            font=dict(size=14)
        ),
        bargap=0.2,
        margin=dict(t=30, b=20, l=0, r=0),
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
    'Estéril': '#AAAAAA',
    'LG': '#4CAF50',
    'MG': '#FFC300',
    'HG': '#FF5733',
    'HL': '#2D3D70'
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
    tipo_agregacao='sum',
    colunas_texto=['Justificativa Alimentação Britagem']
)

grafico_barra_britagem = gerar_grafico_colunas(
    df_agrupado=df_agg_britagem,
    valor_referencia=310,  
    titulo='Alimentação Britagem (t)',
    yaxis_min=0,
    yaxis_max=470,
    colunas_tooltip=['Justificativa Alimentação Britagem']
)

# Grafico 2 - Alimentação Moagem
df_agg_moagem = agregar_por_hora(
    df=df_base_planta,
    coluna_hora='Timestamp',
    valor_coluna='Moinho_Massa Alimentada Moagem_(t)',
    tipo_agregacao='sum',
    colunas_texto=['Justificativa Alimentação Moagem','Desvio taxa Moagem']
)

grafico_barra_moagem = gerar_grafico_colunas(
    df_agrupado=df_agg_moagem,
    valor_referencia=250,  
    titulo='Alimentação Moagem (t)',
    yaxis_min=0,
    yaxis_max=470,
    colunas_tooltip=['Justificativa Alimentação Moagem','Desvio taxa Moagem']
)

# =========================================================
# Funções para Calculos de Ritmo, Produção Acumulada e Ritmo de Produção
# =========================================================

# Função para calcular o acumulado mensal
def acumulado_mensal(
    df: pd.DataFrame,
    coluna_valor: str,
    coluna_datahora: str,
    tipo_agregacao: str = 'sum'
) -> float:
    # Fuso horário local
    fuso = pytz.timezone('America/Sao_Paulo')
    agora = datetime.now(fuso)

    # Se for dia 1, usar o mês anterior como base
    data_base = agora - timedelta(days=1) if agora.day == 1 else agora
    mes = data_base.month
    ano = data_base.year

    # Garantir que a coluna de data esteja em datetime com timezone correto
    try:
        df[coluna_datahora] = pd.to_datetime(df[coluna_datahora])
        if df[coluna_datahora].dt.tz is None:
            # Timestamps são tz-naive → localizar com o fuso de São Paulo
            df[coluna_datahora] = df[coluna_datahora].dt.tz_localize(fuso)
        else:
            # Timestamps já têm timezone → converter para São Paulo
            df[coluna_datahora] = df[coluna_datahora].dt.tz_convert(fuso)
    except Exception as e:
        raise ValueError(f"Erro ao processar a coluna '{coluna_datahora}' como datetime: {e}")

    # Filtrar os dados para o mês e ano desejado
    df_filtrado = df[
        (df[coluna_datahora].dt.month == mes) &
        (df[coluna_datahora].dt.year == ano)
    ]

    # Agregação
    if tipo_agregacao == 'sum':
        return df_filtrado[coluna_valor].sum()
    elif tipo_agregacao == 'mean':
        return df_filtrado[coluna_valor].mean()
    elif tipo_agregacao == 'max':
        return df_filtrado[coluna_valor].max()
    elif tipo_agregacao == 'min':
        return df_filtrado[coluna_valor].min()
    elif tipo_agregacao == 'count':
        return df_filtrado[coluna_valor].count()
    else:
        raise ValueError(f"Tipo de agregação '{tipo_agregacao}' não suportado.")
    
# Função para calcular o valor do dia anterior
def acumulado_dia_anterior(
    df: pd.DataFrame,
    coluna_valor: str,
    coluna_datahora: str,
    tipo_agregacao: str = 'sum'
) -> float:
    agora = datetime.now(pytz.timezone('America/Sao_Paulo'))
    ontem = (agora - timedelta(days=1)).date()

    df[coluna_datahora] = pd.to_datetime(df[coluna_datahora]).dt.tz_convert('America/Sao_Paulo')
    df['data'] = df[coluna_datahora].dt.date
    df_filtrado = df[df['data'] == ontem]

    if tipo_agregacao == 'sum':
        return df_filtrado[coluna_valor].sum()
    elif tipo_agregacao == 'mean':
        return df_filtrado[coluna_valor].mean()
    elif tipo_agregacao == 'max':
        return df_filtrado[coluna_valor].max()
    elif tipo_agregacao == 'min':
        return df_filtrado[coluna_valor].min()
    elif tipo_agregacao == 'count':
        return df_filtrado[coluna_valor].count()
    else:
        raise ValueError(f"Tipo de agregação '{tipo_agregacao}' não suportado.")
    
# Função para calcular o acumulado do dia atual
def acumulado_dia_atual(
    df: pd.DataFrame,
    coluna_valor: str,
    coluna_datahora: str,
    tipo_agregacao: str = 'sum'
) -> float:
    agora = datetime.now(pytz.timezone('America/Sao_Paulo'))
    hoje = agora.date()

    df[coluna_datahora] = pd.to_datetime(df[coluna_datahora]).dt.tz_convert('America/Sao_Paulo')
    df['data'] = df[coluna_datahora].dt.date
    df_filtrado = df[df['data'] == hoje]

    if tipo_agregacao == 'sum':
        return df_filtrado[coluna_valor].sum()
    elif tipo_agregacao == 'mean':
        return df_filtrado[coluna_valor].mean()
    elif tipo_agregacao == 'max':
        return df_filtrado[coluna_valor].max()
    elif tipo_agregacao == 'min':
        return df_filtrado[coluna_valor].min()
    elif tipo_agregacao == 'count':
        return df_filtrado[coluna_valor].count()
    else:
        raise ValueError(f"Tipo de agregação '{tipo_agregacao}' não suportado.")

# Função para calcular o ritmo de produção
def ritmo_mensal(
    df: pd.DataFrame,
    coluna_valor: str,
    coluna_datahora: str,
    tipo_agregacao: str = 'sum'
) -> float:
    fuso = pytz.timezone('America/Sao_Paulo')
    agora = datetime.now(fuso)

    if agora.day == 1:
        data_base = agora - timedelta(days=1)
    else:
        data_base = agora

    mes = data_base.month
    ano = data_base.year

    # Corrigir datetime com fuso horário (atenção aqui!)
    df[coluna_datahora] = pd.to_datetime(df[coluna_datahora])
    if df[coluna_datahora].dt.tz is None:
        df[coluna_datahora] = df[coluna_datahora].dt.tz_localize(fuso)
    else:
        df[coluna_datahora] = df[coluna_datahora].dt.tz_convert(fuso)

    df_mes = df[
        (df[coluna_datahora].dt.month == mes) &
        (df[coluna_datahora].dt.year == ano)
    ]

    if df_mes.empty:
        return 0.0

    if tipo_agregacao == 'sum':
        acumulado = df_mes[coluna_valor].sum()
    elif tipo_agregacao == 'mean':
        acumulado = df_mes[coluna_valor].mean()
    elif tipo_agregacao == 'max':
        acumulado = df_mes[coluna_valor].max()
    elif tipo_agregacao == 'min':
        acumulado = df_mes[coluna_valor].min()
    elif tipo_agregacao == 'count':
        acumulado = df_mes[coluna_valor].count()
    else:
        raise ValueError(f"Tipo de agregação '{tipo_agregacao}' não suportado.")

    data_min = df_mes[coluna_datahora].min()
    agora = datetime.now(fuso).replace(minute=0, second=0, microsecond=0)
    data_max = pd.Timestamp(agora)

    horas_decorridas = int((data_max - data_min).total_seconds() // 3600)

    inicio_mes = pd.Timestamp(datetime(ano, mes, 1), tz=fuso)
    inicio_mes_proximo = (inicio_mes + pd.offsets.MonthBegin(1))
    fim_mes = inicio_mes_proximo - timedelta(seconds=1)
    total_horas_mes = int((fim_mes - inicio_mes).total_seconds() // 3600) + 1

    if total_horas_mes - horas_decorridas == 0:
        return acumulado

    ritmo = ((acumulado / horas_decorridas) * (total_horas_mes - horas_decorridas)) + acumulado
    return ritmo

# Função para calcular o ritmo do dia atual
def ritmo_dia_atual(
    df: pd.DataFrame,
    coluna_valor: str,
    coluna_datahora: str,
    tipo_agregacao: str = 'sum'
) -> float:
    fuso = pytz.timezone('America/Sao_Paulo')
    agora = datetime.now(fuso).replace(minute=0, second=0, microsecond=0)
    hoje = agora.date()

    # Padroniza datetime com fuso horário
    df[coluna_datahora] = pd.to_datetime(df[coluna_datahora])

    if df[coluna_datahora].dt.tz is None:
        df[coluna_datahora] = df[coluna_datahora].dt.tz_localize(fuso)
    else:
        df[coluna_datahora] = df[coluna_datahora].dt.tz_convert(fuso)

    # Filtra apenas registros do dia atual
    df_dia = df[
    (df[coluna_datahora].dt.date == hoje) &
    (df[coluna_datahora].dt.hour < agora.hour)
]

    if df_dia.empty:
        return 0.0

    # Cálculo do acumulado conforme tipo de agregação
    if tipo_agregacao == 'sum':
        acumulado = df_dia[coluna_valor].sum()
    elif tipo_agregacao == 'mean':
        acumulado = df_dia[coluna_valor].mean()
    elif tipo_agregacao == 'max':
        acumulado = df_dia[coluna_valor].max()
    elif tipo_agregacao == 'min':
        acumulado = df_dia[coluna_valor].min()
    elif tipo_agregacao == 'count':
        acumulado = df_dia[coluna_valor].count()
    else:
        raise ValueError(f"❌ Tipo de agregação '{tipo_agregacao}' não suportado.")

    # Considera 00:00 do dia atual como início
    inicio_dia = datetime.combine(hoje, datetime.min.time()).replace(tzinfo=fuso)
    fim_dia = datetime.combine(hoje, datetime.min.time()).replace(tzinfo=fuso) + pd.Timedelta(hours=24)


    #horas_decorridas = int((agora - inicio_dia).total_seconds() // 3600)
    horas_decorridas = horas_decorridas = agora.hour
    total_horas_dia = int((fim_dia - inicio_dia).total_seconds() // 3600)

    if horas_decorridas == 0 or total_horas_dia - horas_decorridas == 0:
        return acumulado

    ritmo = ((acumulado / horas_decorridas) * (total_horas_dia - horas_decorridas)) + acumulado
    return ritmo

# =============================================
# Chamada das funções de cálculo
# =============================================

# Chamada das função de Acumulado do mês
# =============================================

# 1 - Acumulado Movimentação mina do mês
valor_mensal_movimentacao_mina = acumulado_mensal(
    df=df_transporte_filtrado,
    coluna_valor='calculated_mass',
    coluna_datahora='hora_completa',
    tipo_agregacao='sum'
)

# 2 - Acumulado Viagens mina do mês
valor_mensal_viagens = acumulado_mensal(
    df=df_transporte_filtrado,
    coluna_valor='calculated_mass',
    coluna_datahora='hora_completa',
    tipo_agregacao='count'
)

# 3 - Acumulado Britagem do mês
valor_mensal_britagem = acumulado_mensal(
    df=df_dados_planta,
    coluna_valor='Britagem_Massa Produzida Britagem_(t)',
    coluna_datahora='Timestamp',
    tipo_agregacao='sum'
)

# 4 - Acumulado Moagem do mês
valor_mensal_moagem = acumulado_mensal(
    df=df_dados_planta,
    coluna_valor='Moinho_Massa Alimentada Moagem_(t)',
    coluna_datahora='Timestamp',
    tipo_agregacao='sum'
)

# Chamada das função de Ritmo do mês
# ===================================

# 1 - Ritmo Britagem do mês
ritmo_movimentacao = ritmo_mensal(
    df=df_transporte_filtrado,
    coluna_valor='calculated_mass',
    coluna_datahora='hora_completa',
    tipo_agregacao='sum'
)

# 2 - Ritmo Britagem do mês
ritmo_viagens = ritmo_mensal(
    df=df_transporte_filtrado,
    coluna_valor='calculated_mass',
    coluna_datahora='hora_completa',
    tipo_agregacao='count'
)

# 3 - Ritmo Britagem do mês
ritmo_britagem = ritmo_mensal(
    df=df_dados_planta,
    coluna_valor='Britagem_Massa Produzida Britagem_(t)',
    coluna_datahora='Timestamp',
    tipo_agregacao='sum'
)

# 4 - Ritmo Moagem do mês
ritmo_moagem = ritmo_mensal(
    df=df_dados_planta,
    coluna_valor='Moinho_Massa Alimentada Moagem_(t)',
    coluna_datahora='Timestamp',
    tipo_agregacao='sum'
)

# Chamada das função dia anterior
# ================================

# 1 - Dia anterior movimentação de mina
valor_ontem_movimentacao = acumulado_dia_anterior(
    df=df_transporte_filtrado,
    coluna_valor='calculated_mass',
    coluna_datahora='hora_completa',
    tipo_agregacao='sum'
)

# 2 - Dia anterior movimentação de mina
valor_ontem_viagens= acumulado_dia_anterior(
    df=df_transporte_filtrado,
    coluna_valor='calculated_mass',
    coluna_datahora='hora_completa',
    tipo_agregacao='count'
)

# 3 - Dia anterior Britagem
valor_ontem_britagem = acumulado_dia_anterior(
    df=df_dados_planta,
    coluna_valor='Britagem_Massa Produzida Britagem_(t)',
    coluna_datahora='Timestamp',
    tipo_agregacao='sum'
)

# 4 - Dia anterior Britagem
valor_ontem_moagem = acumulado_dia_anterior(
    df=df_dados_planta,
    coluna_valor='Moinho_Massa Alimentada Moagem_(t)',
    coluna_datahora='Timestamp',
    tipo_agregacao='sum'
)

# Chamada das função soma do dia Atual
# =====================================

# 1 - Dia atual movimentação de mina
valor_hoje_movimentacao = acumulado_dia_atual(
    df=df_transporte_filtrado,
    coluna_valor='calculated_mass',
    coluna_datahora='hora_completa',
    tipo_agregacao='sum'
)

# 2 - Dia atual movimentação de mina
valor_hoje_viagens= acumulado_dia_atual(
    df=df_transporte_filtrado,
    coluna_valor='calculated_mass',
    coluna_datahora='hora_completa',
    tipo_agregacao='count'
)

# 3 - Dia atual Britagem
valor_hoje_britagem = acumulado_dia_atual(
    df=df_dados_planta,
    coluna_valor='Britagem_Massa Produzida Britagem_(t)',
    coluna_datahora='Timestamp',
    tipo_agregacao='sum'
)

# 4 - Dia atual Britagem
valor_hoje_moagem = acumulado_dia_atual(
    df=df_dados_planta,
    coluna_valor='Moinho_Massa Alimentada Moagem_(t)',
    coluna_datahora='Timestamp',
    tipo_agregacao='sum'
)

# Chamada das funções de Ritmo do dia atual
# ==========================================

# 1 - Ritmo Britagem do dia
ritmo_movimentacao_dia = ritmo_dia_atual(
    df=df_transporte_filtrado,
    coluna_valor='calculated_mass',
    coluna_datahora='hora_completa',
    tipo_agregacao='sum'
)

# 2 - Ritmo Britagem do dia
ritmo_viagens_dia = ritmo_dia_atual(
    df=df_transporte_filtrado,
    coluna_valor='calculated_mass',
    coluna_datahora='hora_completa',
    tipo_agregacao='count'
)

# 3 - Ritmo Britagem do dia
ritmo_britagem_dia = ritmo_dia_atual(
    df=df_dados_planta,
    coluna_valor='Britagem_Massa Produzida Britagem_(t)',
    coluna_datahora='Timestamp',
    tipo_agregacao='sum'
)

# 4 - Ritmo Moagem do dia
ritmo_moagem_dia = ritmo_dia_atual(
    df=df_dados_planta,
    coluna_valor='Moinho_Massa Alimentada Moagem_(t)',
    coluna_datahora='Timestamp',
    tipo_agregacao='sum'
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
            padding-top: 0rem !important;
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

# Carregamento dos icones
#=========================

#caminhos das imagens
pasta_atual = os.path.dirname(__file__)
logo_aura = os.path.join(pasta_atual, "Icones", "Logo_Aura.jpg")
logo_mina = os.path.join(pasta_atual, "Icones", "caminhao.png")
logo_moagem = os.path.join(pasta_atual, "Icones", "mill.png")
logo_kpi = os.path.join(pasta_atual, "Icones", "kpi2.png")

# Função para converter imagem em base64 e obter o tipo MIME
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
base64_kpi, tipo_kpi = imagem_para_base64_e_tipo(logo_kpi)

# Funções para Exibição de KPIs Customizados
#============================================

# Função para exibir KPIs customizados
def exibir_kpis_customizados(
    valores: dict,
    imagem_base64: str = None,
    imagem_tipo: str = None,
    cor_valor: str = "#2D3D70",
    cor_label: str = "#555",
    fonte_valor: str = "22px",
    fonte_label: str = "16px",
    altura_imagem: str = "64px",
    largura_imagem: str = "64px",
    margin_top: str = "0px",
    margin_bottom: str = "10px",
    padding_top_imagem: str = "0px"  # << novo parâmetro para ajustar posição vertical da imagem
):
    # Duas colunas principais: imagem (esquerda) + kpis (direita)
    col_img, col_kpis = st.columns([1, 6])  # Ajuste a proporção se necessário

    # Bloco da imagem única
    if imagem_base64 and imagem_tipo:
        with col_img:
            st.markdown(
                f"""
                <div style="display: flex; justify-content: center; padding-top: {padding_top_imagem}; margin-top: {margin_top}; margin-bottom: {margin_bottom};">
                    <img src="data:{imagem_tipo};base64,{imagem_base64}"
                         style="height: {altura_imagem}; width: {largura_imagem}; object-fit: contain;" />
                </div>
                """,
                unsafe_allow_html=True
            )

    # Bloco dos KPIs
    with col_kpis:
        num_kpis = len(valores)
        colunas = st.columns(num_kpis)

        for i, (label, valor) in enumerate(valores.items()):
            with colunas[i]:
                html = f"""
                    <div style="text-align: left; margin-top: {margin_top}; margin-bottom: {margin_bottom}; line-height: 1.2;">
                        <span style="font-size:{fonte_label}; color:{cor_label}; white-space: nowrap;">{label}</span><br>
                        <b style="font-size:{fonte_valor}; color:{cor_valor}; white-space: nowrap;">{valor:,.0f}</b>
                    </div>
                """
                st.markdown(html, unsafe_allow_html=True)


# ==================================================
# Renderização do Dashboard em Tela Única (Full HD)
# ==================================================

# Cabeçalho Mina
st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center;
        background-color: #2D3D70; padding: 0px 30px; border-radius: 8px; margin-top: 0.2px; margin-bottom: 0.2px;">
        <img src="data:{tipo_esquerda};base64,{base64_esquerda}" style="height: 45px;">
        <h1 style="color: white; font-size: 28px; margin: 0;">Performance Mina Paiol - Aura Almas</h1>
        <img src="data:{tipo_direita};base64,{base64_direita}" style="height: 40px;">
    </div>
""", unsafe_allow_html=True)

# Criação do Layout de cada linha
#=================================

# Linha 1 - Movimentação Total / Numero de Viagens
col1, col2 = st.columns([0.5, 0.5], gap="large")
with col1:
    valores_kpis = {
        "Acumulado": valor_mensal_viagens,
        "Ritmo Mês": ritmo_viagens,
        "Ontem": valor_ontem_viagens,
        "Hoje": valor_hoje_viagens,
        "Ritmo Dia": ritmo_viagens_dia,
        #"Meta dia": 710
    }

    exibir_kpis_customizados(
        valores=valores_kpis,
        imagem_base64=base64_kpi,
        imagem_tipo=tipo_kpi,
        cor_valor="#2D3D70",
        cor_label="#444",
        fonte_valor="22px",
        fonte_label="14px",
        altura_imagem="26px",
        margin_top="0px",
        margin_bottom="10px",
        padding_top_imagem="15px"
    )
    if not df_agg_viagens.empty:
        st.plotly_chart(grafico_numero_viagens.update_layout(height=270), use_container_width=True)

with col2:
    valores_kpis = {
        "Acumulado": valor_mensal_movimentacao_mina,
        "Ritmo Mês": ritmo_movimentacao,
        "Ontem": valor_ontem_movimentacao,
        "Hoje": valor_hoje_movimentacao,
        "Ritmo Dia": ritmo_movimentacao_dia,
        #"Meta dia": 71000
    }

    exibir_kpis_customizados(
        valores=valores_kpis,
        imagem_base64=base64_kpi,
        imagem_tipo=tipo_kpi,
        cor_valor="#2D3D70",
        cor_label="#444",
        fonte_valor="22px",
        fonte_label="14px",
        altura_imagem="26px",
        margin_top="0px",
        margin_bottom="10px",
        padding_top_imagem="15px"
    )
    if not df_agg_viagens.empty:
        st.plotly_chart(grafico_movimentacao_litogia.update_layout(height=270), use_container_width=True)

# Cabeçalho Moagem
st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center;
        background-color: #2D3D70; padding: 0px 30px; border-radius: 8px; margin-top: 70px; margin-bottom: 5px;">
        <img src="data:{tipo_esquerda2};base64,{base64_esquerda2}" style="height: 40px;">
        <h1 style="color: white; font-size: 28px; margin: 0;">Performance Planta - Aura Almas</h1>
        <img src="data:{tipo_direita};base64,{base64_direita}" style="height: 40px;">
    </div>
""", unsafe_allow_html=True)

# Linha 2 - Alimentação Britagem / Alimentação Moagem
col3, col4 = st.columns([0.5, 0.5], gap="large")
with col3:
    valores_kpis = {
        "Acumulado": valor_mensal_britagem,
        "Ritmo Mês": ritmo_britagem,
        "Ontem": valor_ontem_britagem,
        "Hoje": valor_hoje_britagem,
        "Ritmo Dia": ritmo_britagem_dia,
        #"Meta dia": 71000
    }

    exibir_kpis_customizados(
        valores=valores_kpis,
        imagem_base64=base64_kpi,
        imagem_tipo=tipo_kpi,
        cor_valor="#2D3D70",
        cor_label="#444",
        fonte_valor="22px",
        fonte_label="14px",
        altura_imagem="26px",
        margin_top="0px",
        margin_bottom="10px",
        padding_top_imagem="15px"
    )
    if not df_agg_britagem.empty:
        st.plotly_chart(grafico_barra_britagem.update_layout(height=270), use_container_width=True)
with col4:
    valores_kpis = {
        "Acumulado": valor_mensal_moagem,
        "Ritmo Mês": ritmo_moagem,
        "Ontem": valor_ontem_moagem,
        "Hoje": valor_hoje_moagem,
        "Ritmo Dia": ritmo_moagem_dia,
        #"Meta dia": 71000
    }

    exibir_kpis_customizados(
        valores=valores_kpis,
        imagem_base64=base64_kpi,
        imagem_tipo=tipo_kpi,
        cor_valor="#2D3D70",
        cor_label="#444",
        fonte_valor="22px",
        fonte_label="14px",
        altura_imagem="26px",
        margin_top="0px",
        margin_bottom="10px",
        padding_top_imagem="15px"
    )
    if not df_agg_moagem.empty:
        st.plotly_chart(grafico_barra_moagem.update_layout(height=270), use_container_width=True)