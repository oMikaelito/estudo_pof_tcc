import io

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Neuroeconomia | Ultraprocessados POF", layout="wide")

COOL_PALETTE = ["#B07CE0", "#7B2CBF", "#C9A7EB", "#4A148C"]  # roxo/lilás
ANO_COLOR_MAP = {"2008": COOL_PALETTE[0], "2018": COOL_PALETTE[3]}
VALOR_COLORSCALE = ["#E4D3F7", "#9D4EDD", "#3C096C"]  # claro (menor gasto) -> escuro (maior gasto)
POF_LABEL_MAP = {2008: "POF 2008-2009", 2018: "POF 2017-2018"}  # só para exibição
TEXT_COLOR = "#2B2B2B"
GRID_COLOR = "#E5E5E5"

# Esquema de cores das tabelas (roxo/lilás)
TABLE_TEXT_COLOR = "#3B1F5C"      # roxo escuro
TABLE_ROW_ODD = "#F1EAFB"         # lilás claro
TABLE_ROW_EVEN = "#E2D4F5"        # lilás um pouco mais forte

REQUIRED_COLUMNS = ["Ano_Pesquisa", "Produto", "Categoria", "Gasto_Reais"]


def style_table_purple(df: pd.DataFrame):
    """Aplica o esquema roxo/lilás (linhas zebradas + texto roxo escuro) a uma tabela."""
    def _colors(data: pd.DataFrame) -> pd.DataFrame:
        styles = pd.DataFrame("", index=data.index, columns=data.columns)
        for i in range(len(data)):
            bg = TABLE_ROW_ODD if i % 2 == 0 else TABLE_ROW_EVEN
            styles.iloc[i, :] = f"background-color: {bg}; color: {TABLE_TEXT_COLOR}"
        return styles

    return df.style.apply(_colors, axis=None)


def format_brl(value: float) -> str:
    sinal = "-" if value < 0 else ""
    texto = f"{abs(value):,.2f}"
    texto = texto.replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{sinal}R$ {texto}"


def format_pct(value: float) -> str:
    return f"{value:+.1f}%".replace(".", ",")


def parse_currency_value(value) -> float:
    if pd.isna(value):
        return np.nan

    text = str(value).strip()
    text = text.replace("R$", "").replace(" ", "")

    if text == "":
        return np.nan

    if "," in text:
        text = text.replace(".", "").replace(",", ".")

    try:
        return float(text)
    except ValueError:
        return np.nan


def convert_gasto_column(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    df = df.copy()
    df["Gasto_Reais"] = df["Gasto_Reais"].apply(parse_currency_value)

    linhas_invalidas = df["Gasto_Reais"].isna().sum()
    df = df.dropna(subset=["Gasto_Reais"])

    return df, int(linhas_invalidas)


def convert_ano_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Ano_Pesquisa"] = pd.to_numeric(df["Ano_Pesquisa"], errors="coerce")
    df = df.dropna(subset=["Ano_Pesquisa"])
    df["Ano_Pesquisa"] = df["Ano_Pesquisa"].astype(int)
    return df


def strip_text_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    df = df.copy()
    for coluna in ["Produto", "Categoria", "Classificacao"]:
        if coluna in df.columns:
            df[coluna] = df[coluna].astype("string").str.strip().replace("", pd.NA)

    linhas_antes = len(df)
    colunas_obrigatorias_texto = [c for c in ["Produto", "Categoria"] if c in df.columns]
    if colunas_obrigatorias_texto:
        df = df.dropna(subset=colunas_obrigatorias_texto)
    linhas_sem_produto_categoria = linhas_antes - len(df)

    if "Classificacao" in df.columns:
        df["Classificacao"] = df["Classificacao"].fillna("Não informada")

    return df, linhas_sem_produto_categoria


def count_duplicate_rows(df: pd.DataFrame) -> int:
    duplicadas = df.duplicated(subset=["Ano_Pesquisa", "Produto"], keep=False)
    return int(duplicadas.sum())


def find_products_in_single_year(df: pd.DataFrame) -> list:
    anos_por_produto = df.groupby("Produto")["Ano_Pesquisa"].nunique()
    produtos_um_ano = anos_por_produto[anos_por_produto < 2].index.tolist()
    return sorted(produtos_um_ano)


def validate_required_columns(df: pd.DataFrame) -> list:
    return [col for col in REQUIRED_COLUMNS if col not in df.columns]


def decode_with_fallback(raw_bytes: bytes) -> str:
    try:
        return raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw_bytes.decode("latin-1")


def detect_csv_separator(text_sample: str) -> str:
    primeira_linha = text_sample.split("\n")[0]
    return ";" if primeira_linha.count(";") > primeira_linha.count(",") else ","


def load_uploaded_file(uploaded_file) -> pd.DataFrame:
    file_name = uploaded_file.name.lower()

    if file_name.endswith(".xlsx"):
        df = pd.read_excel(io.BytesIO(uploaded_file.getvalue()))
    else:
        raw_bytes = uploaded_file.getvalue()
        text = decode_with_fallback(raw_bytes)
        sep = detect_csv_separator(text)
        df = pd.read_csv(io.StringIO(text), sep=sep)

    df.columns = [str(c).strip() for c in df.columns]
    return df


def process_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, list, list]:
    linhas_lidas = len(df)
    df = df.dropna(how="all")

    missing_cols = validate_required_columns(df)
    if missing_cols:
        colunas_encontradas = df.columns.tolist()
        return df, {}, missing_cols, colunas_encontradas

    df, linhas_sem_produto_categoria = strip_text_columns(df)

    df, linhas_invalidas_gasto = convert_gasto_column(df)

    linhas_antes_ano = len(df)
    df = convert_ano_column(df)
    linhas_ano_nao_numerico = linhas_antes_ano - len(df)

    linhas_antes_ano_aceito = len(df)
    anos_inesperados = sorted(int(ano) for ano in set(df["Ano_Pesquisa"].unique()) - {2008, 2018})
    df = df[df["Ano_Pesquisa"].isin([2008, 2018])]
    linhas_ano_nao_aceito = linhas_antes_ano_aceito - len(df)

    linhas_duplicadas = count_duplicate_rows(df)
    produtos_um_ano = find_products_in_single_year(df)

    resumo = {
        "linhas_lidas": linhas_lidas,
        "linhas_ignoradas_produto_categoria": linhas_sem_produto_categoria,
        "linhas_ignoradas_gasto": linhas_invalidas_gasto,
        "linhas_ignoradas_ano_nao_numerico": linhas_ano_nao_numerico,
        "linhas_ignoradas_ano_nao_aceito": linhas_ano_nao_aceito,
        "linhas_duplicadas": linhas_duplicadas,
        "linhas_finais": len(df),
        "anos_encontrados": sorted(df["Ano_Pesquisa"].unique().tolist()),
        "anos_inesperados": anos_inesperados,
        "produtos_um_ano": produtos_um_ano,
    }

    return df, resumo, [], []


def show_import_summary(resumo: dict) -> None:
    with st.expander("Resumo da importação", expanded=True):
        st.write(f"Linhas lidas da planilha: {resumo['linhas_lidas']}")
        st.write(f"Linhas usadas no final: {resumo['linhas_finais']}")

        if resumo["linhas_ignoradas_produto_categoria"] > 0:
            st.write(
                f"Linhas ignoradas por estarem sem Produto ou Categoria preenchidos: "
                f"{resumo['linhas_ignoradas_produto_categoria']}"
            )

        if resumo["linhas_ignoradas_gasto"] > 0:
            st.write(
                f"Linhas ignoradas por o valor gasto não ser um número reconhecido: "
                f"{resumo['linhas_ignoradas_gasto']}"
            )

        if resumo["linhas_ignoradas_ano_nao_numerico"] > 0:
            st.write(
                f"Linhas ignoradas por o ano da pesquisa não ser um número reconhecido: "
                f"{resumo['linhas_ignoradas_ano_nao_numerico']}"
            )

        if resumo["linhas_ignoradas_ano_nao_aceito"] > 0:
            st.write(
                f"Linhas ignoradas por terem um ano diferente de 2008 ou 2018: "
                f"{resumo['linhas_ignoradas_ano_nao_aceito']}"
            )

        st.write(f"Anos encontrados na planilha (já filtrados): {resumo['anos_encontrados']}")

        if resumo["anos_inesperados"]:
            st.warning(
                f"A planilha tinha ano(s) diferente(s) de 2008 e 2018: {resumo['anos_inesperados']}. "
                "Essas linhas foram desconsideradas, pois só 2008 e 2018 são aceitos."
            )

        if resumo["linhas_duplicadas"] > 0:
            st.warning(
                f"{resumo['linhas_duplicadas']} linha(s) têm o mesmo Produto e Ano_Pesquisa. "
                "Confira se não há linhas duplicadas na planilha (elas não foram removidas)."
            )

        if resumo["produtos_um_ano"]:
            st.warning(
                f"{len(resumo['produtos_um_ano'])} produto(s) aparecem em apenas um dos anos. "
                "Isso afeta a variação total, já que não há como comparar esses produtos entre 2008 e 2018. "
                "Isso pode acontecer porque algumas linhas foram descartadas na importação (veja o resumo acima) "
                "ou porque o nome do produto está escrito de forma diferente nos dois anos."
            )


def show_single_year_products_expander(produtos_um_ano: list) -> None:
    if not produtos_um_ano:
        return
    with st.expander("Ver produtos que aparecem em apenas um ano"):
        for produto in produtos_um_ano:
            st.write(f"- {produto}")


@st.cache_data
def load_uploaded_data(file_bytes: bytes, file_name: str) -> tuple[pd.DataFrame, dict, list, list]:
    class _BufferedFile:
        def __init__(self, data, name):
            self._data = data
            self.name = name

        def getvalue(self):
            return self._data

    buffered = _BufferedFile(file_bytes, file_name)
    df_raw = load_uploaded_file(buffered)
    return process_dataframe(df_raw)


def show_upload_guide() -> None:
    st.info(
        "Envie sua planilha pela barra lateral (.csv ou .xlsx).\n\n"
        "Deve ser uma única planilha com os dois anos juntos: uma linha por produto por ano."
    )

    st.markdown(
        "**Colunas esperadas:**\n"
        "- `Ano_Pesquisa` (2008 ou 2018)\n"
        "- `Produto`\n"
        "- `Categoria`\n"
        "- `Gasto_Reais` — só números, sem \"R$\"; vírgula ou ponto decimal são aceitos\n"
        "- `Classificacao` (opcional)\n\n"
        "Se o arquivo for pesado, o formato CSV é recomendado."
    )

    exemplo = pd.DataFrame({
        "Ano_Pesquisa": [2008, 2008, 2018, 2018],
        "Produto": ["Refrigerante Cola 2L", "Salsicha", "Refrigerante Cola 2L", "Salsicha"],
        "Categoria": ["Refrigerantes", "Embutidos", "Refrigerantes", "Embutidos"],
        "Gasto_Reais": ["45,50", "18,90", "62,30", "24,10"],
    })
    st.caption("Exemplo ilustrativo (não entra em nenhum cálculo):")
    st.dataframe(style_table_purple(exemplo), use_container_width=True, hide_index=True)


def build_variation_table(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    pivot = df.groupby([group_col, "Ano_Pesquisa"])["Gasto_Reais"].sum().unstack("Ano_Pesquisa")
    pivot = pivot.reindex(columns=[2008, 2018])

    if group_col == "Produto":
        pivot = pivot.dropna(subset=[2008, 2018])
    else:
        pivot = pivot.fillna(0.0)

    pivot = pivot.rename(columns={2008: "Gasto 2008", 2018: "Gasto 2018"})
    pivot["Variação (R$)"] = pivot["Gasto 2018"] - pivot["Gasto 2008"]
    pivot["Variação (%)"] = np.where(
        pivot["Gasto 2008"] > 0,
        (pivot["Variação (R$)"] / pivot["Gasto 2008"]) * 100,
        np.nan
    )
    pivot = pivot.sort_values("Variação (%)", ascending=False, na_position="last")
    pivot = pivot.reset_index()
    return pivot


def show_variation_table(pivot: pd.DataFrame) -> None:
    if pivot.empty:
        st.info("Nenhum dado disponível para esta tabela com os filtros atuais.")
        return

    display_df = pivot.copy()
    display_df["Gasto 2008"] = display_df["Gasto 2008"].apply(format_brl)
    display_df["Gasto 2018"] = display_df["Gasto 2018"].apply(format_brl)
    display_df["Variação (R$)"] = display_df["Variação (R$)"].apply(format_brl)
    display_df["Variação (%)"] = display_df["Variação (%)"].apply(
        lambda v: format_pct(v) if pd.notna(v) else "—"
    )
    st.dataframe(style_table_purple(display_df), use_container_width=True, hide_index=True)


def build_share_chart(df_base_classificacao: pd.DataFrame, categorias_selecionadas: list) -> go.Figure:
    totals = df_base_classificacao.groupby(["Ano_Pesquisa", "Categoria"])["Gasto_Reais"].sum().reset_index()
    totals_por_ano = totals.groupby("Ano_Pesquisa")["Gasto_Reais"].transform("sum")
    totals["Participacao"] = (totals["Gasto_Reais"] / totals_por_ano) * 100
    totals["Ano_Pesquisa"] = totals["Ano_Pesquisa"].astype(str)

    totals = totals[totals["Categoria"].isin(categorias_selecionadas)]

    fig = px.bar(
        totals,
        x="Categoria",
        y="Participacao",
        color="Ano_Pesquisa",
        barmode="group",
        color_discrete_map=ANO_COLOR_MAP,
        text="Participacao",
        title="Participação (%) de Cada Categoria no Gasto Total do Ano"
    )

    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
        hovertemplate="%{y:,.1f}%<extra></extra>"
    )

    fig.update_layout(
        xaxis_title="Categoria",
        yaxis_title="Participação no Total (%)",
        legend_title=None,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=13, color=TEXT_COLOR),
        title_font=dict(size=18, color=TEXT_COLOR),
        title_x=0,
        legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, font=dict(color=TEXT_COLOR)),
        xaxis_tickangle=-30,
        margin=dict(t=120, b=130, l=80, r=40),
        separators=",.",
        height=480,
        bargap=0.25,
        bargroupgap=0.05
    )
    fig.update_xaxes(
        automargin=True,
        title_font=dict(color=TEXT_COLOR),
        tickfont=dict(color=TEXT_COLOR),
        showgrid=False,
        linecolor=GRID_COLOR
    )
    fig.update_yaxes(
        automargin=True,
        title_standoff=20,
        title_font=dict(color=TEXT_COLOR),
        tickfont=dict(color=TEXT_COLOR),
        showgrid=True,
        gridcolor=GRID_COLOR,
        linecolor=GRID_COLOR
    )

    return fig


def build_total_comparison_chart(df: pd.DataFrame) -> go.Figure:
    totals = df.groupby("Ano_Pesquisa", as_index=False)["Gasto_Reais"].sum()
    totals["Ano_Pesquisa"] = totals["Ano_Pesquisa"].astype(str)

    fig = px.bar(
        totals,
        x="Ano_Pesquisa",
        y="Gasto_Reais",
        color="Ano_Pesquisa",
        color_discrete_map=ANO_COLOR_MAP,
        text="Gasto_Reais",
        title="Gasto Total com Ultraprocessados: POF 2008-2009 vs 2017-2018"
    )

    fig.update_traces(
        texttemplate="R$ %{text:,.2f}",
        textposition="outside",
        marker_line_width=0,
        hovertemplate="R$ %{y:,.2f}<extra></extra>"
    )

    fig.update_layout(
        yaxis_title="Gasto Total (R$)",
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=14, color=TEXT_COLOR),
        title_font=dict(size=20, color=TEXT_COLOR),
        title_x=0,
        margin=dict(t=80, b=60, l=80, r=40),
        separators=",.",
        height=480,
        bargap=0.25
    )
    fig.update_xaxes(
        type="category",
        categoryorder="array",
        categoryarray=[str(ano) for ano in POF_LABEL_MAP],
        tickmode="array",
        tickvals=[str(ano) for ano in POF_LABEL_MAP],
        ticktext=list(POF_LABEL_MAP.values()),
        automargin=True,
        title_font=dict(color=TEXT_COLOR),
        tickfont=dict(color=TEXT_COLOR),
        showgrid=False,
        linecolor=GRID_COLOR
    )
    fig.update_yaxes(
        automargin=True,
        title_standoff=20,
        title_font=dict(color=TEXT_COLOR),
        tickfont=dict(color=TEXT_COLOR),
        showgrid=True,
        gridcolor=GRID_COLOR,
        linecolor=GRID_COLOR
    )

    return fig


def build_category_breakdown_chart(df: pd.DataFrame) -> go.Figure:
    cat_totals = df.groupby(["Categoria", "Ano_Pesquisa"], as_index=False)["Gasto_Reais"].sum()
    cat_totals["Ano_Pesquisa"] = cat_totals["Ano_Pesquisa"].astype(str)

    fig = px.bar(
        cat_totals,
        x="Categoria",
        y="Gasto_Reais",
        color="Ano_Pesquisa",
        barmode="group",
        color_discrete_map=ANO_COLOR_MAP,
        title="Gasto por Categoria de Ultraprocessados"
    )

    fig.update_traces(hovertemplate="R$ %{y:,.2f}<extra></extra>")

    fig.update_layout(
        xaxis_title="Categoria",
        yaxis_title="Gasto Total (R$)",
        legend_title=None,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=13, color=TEXT_COLOR),
        title_font=dict(size=18, color=TEXT_COLOR),
        title_x=0,
        legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, font=dict(color=TEXT_COLOR)),
        xaxis_tickangle=-30,
        margin=dict(t=120, b=130, l=80, r=40),
        separators=",.",
        height=480,
        bargap=0.25,
        bargroupgap=0.05
    )
    fig.update_xaxes(
        automargin=True,
        title_font=dict(color=TEXT_COLOR),
        tickfont=dict(color=TEXT_COLOR),
        showgrid=False,
        linecolor=GRID_COLOR
    )
    fig.update_yaxes(
        automargin=True,
        title_standoff=20,
        title_font=dict(color=TEXT_COLOR),
        tickfont=dict(color=TEXT_COLOR),
        showgrid=True,
        gridcolor=GRID_COLOR,
        linecolor=GRID_COLOR
    )

    return fig


def build_top_products_chart(df: pd.DataFrame, ano: int, top_n: int = 8) -> go.Figure:
    subset = df[df["Ano_Pesquisa"] == ano]
    top_products = subset.groupby("Produto", as_index=False)["Gasto_Reais"].sum()
    top_products = top_products.sort_values("Gasto_Reais", ascending=True).tail(top_n)

    max_valor = top_products["Gasto_Reais"].max() if not top_products.empty else 0
    range_max = max_valor * 1.15 if max_valor > 0 else 1

    fig = px.bar(
        top_products,
        x="Gasto_Reais",
        y="Produto",
        orientation="h",
        title=f"Top {top_n} Produtos Ultraprocessados — {ano}"
    )

    fig.update_traces(
        marker=dict(
            color=top_products["Gasto_Reais"],
            colorscale=VALOR_COLORSCALE,
            cmin=top_products["Gasto_Reais"].min(),
            cmax=top_products["Gasto_Reais"].max(),
            showscale=False
        ),
        texttemplate="R$ %{x:,.2f}",
        textposition="outside",
        cliponaxis=False,
        hovertemplate="R$ %{x:,.2f}<extra></extra>"
    )

    fig.update_layout(
        xaxis_title="Gasto Total (R$)",
        yaxis_title="",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=13, color=TEXT_COLOR),
        title_font=dict(size=18, color=TEXT_COLOR),
        title_x=0,
        margin=dict(t=80, b=40, l=180, r=40),
        separators=",.",
        height=420
    )
    fig.update_xaxes(
        range=[0, range_max],
        title_font=dict(color=TEXT_COLOR),
        tickfont=dict(color=TEXT_COLOR),
        showgrid=True,
        gridcolor=GRID_COLOR,
        linecolor=GRID_COLOR
    )
    fig.update_yaxes(
        automargin=True,
        title_font=dict(color=TEXT_COLOR),
        tickfont=dict(color=TEXT_COLOR),
        showgrid=False,
        linecolor=GRID_COLOR
    )

    return fig


def main():
    st.title("Neuroeconomia do Consumo: Ultraprocessados na POF (IBGE)")
    st.caption("Comparativo entre as Pesquisas de Orçamentos Familiares 2008-2009 e 2017-2018")

    with st.sidebar:
        st.header("Dados")
        uploaded_file = st.file_uploader("Enviar planilha (.csv ou .xlsx)", type=["csv", "xlsx"])

    if uploaded_file is None:
        show_upload_guide()
        st.stop()

    try:
        df, resumo, missing_cols, colunas_encontradas = load_uploaded_data(
            uploaded_file.getvalue(), uploaded_file.name
        )
    except Exception as e:
        st.error("Não consegui ler o arquivo. Confira se ele abre normalmente no Excel e envie de novo.")
        with st.expander("Detalhes técnicos"):
            st.exception(e)
        st.stop()

    if missing_cols:
        st.error(
            f"Faltam as colunas: {', '.join(missing_cols)}. "
            f"Colunas encontradas na planilha: {', '.join(colunas_encontradas)}. "
            "Confira o cabeçalho da planilha e envie de novo."
        )
        st.stop()

    if df.empty:
        st.error(
            "Não encontramos nenhuma linha válida na planilha. "
            "Confira se a coluna Ano_Pesquisa está preenchida com 2008 ou 2018 "
            "e se a coluna Gasto_Reais tem valores numéricos (ex.: 45,50)."
        )
        st.stop()

    show_import_summary(resumo)
    show_single_year_products_expander(resumo["produtos_um_ano"])

    with st.sidebar:
        st.header("Filtros")
        categorias_disponiveis = sorted(df["Categoria"].unique().tolist())
        categorias_selecionadas = st.multiselect(
            "Categorias de Produtos",
            options=categorias_disponiveis,
            default=categorias_disponiveis
        )

        classificacoes_selecionadas = None
        if "Classificacao" in df.columns:
            classificacoes_disponiveis = sorted(df["Classificacao"].dropna().unique().tolist())
            classificacoes_selecionadas = st.multiselect(
                "Classificação (grau de processamento)",
                options=classificacoes_disponiveis,
                default=classificacoes_disponiveis
            )

    sem_selecao = (not categorias_selecionadas) or (
        classificacoes_selecionadas is not None and not classificacoes_selecionadas
    )

    if sem_selecao:
        st.warning("Nenhum dado para exibir. Selecione ao menos uma opção nos filtros da barra lateral.")
        return

    df_base_classificacao = df
    if classificacoes_selecionadas is not None:
        df_base_classificacao = df[df["Classificacao"].isin(classificacoes_selecionadas)]

    df_filtrado = df_base_classificacao[df_base_classificacao["Categoria"].isin(categorias_selecionadas)]

    if df_filtrado.empty:
        st.warning("Nenhum dado para exibir. Selecione ao menos uma opção nos filtros da barra lateral.")
        return

    if df_filtrado["Ano_Pesquisa"].nunique() < 2:
        st.warning("A comparação precisa dos dois anos (2008 e 2018); os filtros atuais deixaram apenas um deles.")

    total_2008 = df_filtrado.loc[df_filtrado["Ano_Pesquisa"] == 2008, "Gasto_Reais"].sum()
    total_2018 = df_filtrado.loc[df_filtrado["Ano_Pesquisa"] == 2018, "Gasto_Reais"].sum()

    tabela_produto = build_variation_table(df_filtrado, "Produto")
    total_2008_comparavel = tabela_produto["Gasto 2008"].sum() if not tabela_produto.empty else 0.0
    total_2018_comparavel = tabela_produto["Gasto 2018"].sum() if not tabela_produto.empty else 0.0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Gasto Total 2008-2009", format_brl(total_2008))
    col2.metric("Gasto Total 2017-2018", format_brl(total_2018))

    if total_2008 == 0 or total_2018 == 0:
        col3.metric("Variação Total", "—")
    else:
        variacao_pct = (total_2018 - total_2008) / total_2008 * 100
        col3.metric("Variação Total", format_pct(variacao_pct))

    if total_2008_comparavel == 0 or total_2018_comparavel == 0:
        col4.metric("Variação comparável", "—")
    else:
        variacao_comparavel_pct = (total_2018_comparavel - total_2008_comparavel) / total_2008_comparavel * 100
        col4.metric(
            "Variação comparável",
            format_pct(variacao_comparavel_pct),
            help=(
                "Considera só os produtos presentes em 2008 e 2018, "
                "por isso pode diferir da Variação Total (que usa todos os produtos de cada ano)."
            )
        )

    st.caption(
        "\"Variação Total\" considera todos os produtos de cada ano. "
        "\"Variação comparável\" considera só os produtos presentes em 2008 e 2018, "
        "por isso os dois valores podem ser diferentes."
    )

    st.plotly_chart(build_total_comparison_chart(df_filtrado), use_container_width=True, theme=None)
    st.plotly_chart(build_category_breakdown_chart(df_filtrado), use_container_width=True, theme=None)
    st.plotly_chart(build_share_chart(df_base_classificacao, categorias_selecionadas), use_container_width=True, theme=None)

    st.subheader("Variação por Categoria")
    tabela_categoria = build_variation_table(df_filtrado, "Categoria")
    show_variation_table(tabela_categoria)

    st.subheader("Variação por Produto")
    st.caption("Somente produtos presentes nos dois anos (2008 e 2018).")
    show_variation_table(tabela_produto)

    tab1, tab2 = st.tabs(["Top Produtos 2008-2009", "Top Produtos 2017-2018"])
    with tab1:
        st.plotly_chart(build_top_products_chart(df_filtrado, 2008), use_container_width=True, theme=None)
    with tab2:
        st.plotly_chart(build_top_products_chart(df_filtrado, 2018), use_container_width=True, theme=None)

    with st.expander("Ver dados brutos"):
        st.dataframe(style_table_purple(df_filtrado), use_container_width=True)


if __name__ == "__main__":
    main()
