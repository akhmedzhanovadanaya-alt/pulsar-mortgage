import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix
from io import BytesIO
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter
import plotly.express as px
from io import BytesIO
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
def create_pdf_report(comparison, optimization_params):

    buffer = BytesIO()

    pdf = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        rightMargin=35,
        leftMargin=35,
        topMargin=35,
        bottomMargin=35
    )

    font_path = "NotoSans-Regular.ttf"
    bold_font_path = "NotoSans-Bold.ttf"

    pdfmetrics.registerFont(
        TTFont("Arial", font_path)
    )

    pdfmetrics.registerFont(
        TTFont("Arial-Bold", bold_font_path)
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontName="Arial-Bold",
        fontSize=20,
        textColor=colors.HexColor("#302B3A"),
        alignment=TA_CENTER,
        spaceAfter=18
    )

    heading_style = ParagraphStyle(
        "ReportHeading",
        parent=styles["Heading2"],
        fontName="Arial-Bold",
        fontSize=12,
        textColor=colors.HexColor("#613C4E"),
        spaceBefore=12,
        spaceAfter=8
    )

    normal_style = ParagraphStyle(
        "ReportNormal",
        parent=styles["Normal"],
        fontName="Arial",
        fontSize=9,
        leading=14
    )

    elements = []

    elements.append(
        Paragraph("ПУЛЬСАР", title_style)
    )

    elements.append(
        Paragraph(
            "Аналитический отчёт по результатам "
            "анализа и оптимизации ипотечного пула",
            normal_style
        )
    )

    elements.append(Spacer(1, 12))

    elements.append(
        Paragraph(
            f"Дата формирования: {datetime.now().strftime('%d.%m.%Y %H:%M')}",
            normal_style
        )
    )

    elements.append(Spacer(1, 15))

    elements.append(
        Paragraph("1. Параметры оптимизации", heading_style)
    )

    elements.append(
        Paragraph(
            f"Снижение стоимости залога: "
            f"{optimization_params['collateral_drop']:.0%}",
            normal_style
        )
    )

    elements.append(
        Paragraph(
            f"Множитель PD: "
            f"{optimization_params['pd_multiplier']:.2f}",
            normal_style
        )
    )

    elements.append(Spacer(1, 15))

    elements.append(
        Paragraph("2. Сравнение ипотечных пулов", heading_style)
    )

    table_data = [
        [
            Paragraph(
                str(column),
                ParagraphStyle(
                    "TableHeader",
                    fontName="Arial-Bold",
                    fontSize=6,
                    leading=8,
                    textColor=colors.white,
                    alignment=TA_CENTER
                )
            )
            for column in comparison.columns
        ]
    ]

    for _, row in comparison.iterrows():
        table_data.append([
            str(value) for value in row
        ])

    table = Table(
        table_data,
        colWidths=[
            105, 65, 75, 75, 75,
            75, 75, 75, 75
        ],
        repeatRows=1
    )

    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0),
         colors.HexColor("#613C4E")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Arial-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Arial"),
        ("FONTSIZE", (0, 0), (-1, -1), 6),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#F2F0EF")]),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))

    elements.append(table)

    elements.append(Spacer(1, 18))

    elements.append(
        Paragraph("3. Методологическая оговорка", heading_style)
    )

    elements.append(
        Paragraph(
            "Результаты получены на основе синтетической "
            "выборки ипотечных кредитов и экспериментальной "
            "модели оптимизации. Они предназначены для "
            "демонстрации возможностей системы и не являются "
            "оценкой реального ипотечного портфеля.",
            normal_style
        )
    )

    pdf.build(elements)

    buffer.seek(0)

    return buffer

DATA_FILE = "mortgage_portfolio.csv"
@st.cache_data
def load_data():
    return pd.read_csv(DATA_FILE)


df = load_data()
def optimize_pool(
    data,
    pool_size=250,
    target_balance=1_250_000_000,
    mode="stress",
    collateral_drop=0.20,
    pd_multiplier=2.0
):

    n = len(data)

    # Целевая функция: минимизация стрессовых ожидаемых потерь
    stress_pd = np.clip(
    data["pd"].to_numpy() * pd_multiplier,
    0,
    1
)

    stressed_property = (
        data["property_value"].to_numpy() * (1 - collateral_drop)
    )

    stressed_recovery = stressed_property * 0.85 * 0.9

    stressed_lgd = (
        1 - stressed_recovery / data["ead"].to_numpy()
    ).clip(0, 1)

    if mode == "random":
        objective = np.random.default_rng(42).random(n)

    elif mode == "quality":
        objective = (
            data["ead"].to_numpy()
            * data["pd"].to_numpy()
            * data["lgd"].to_numpy()
        )

    else:
        objective = (
            data["ead"].to_numpy()
            * stress_pd
            * stressed_lgd
        )

    # Ограничения оптимизации
    matrix = lil_matrix((8, n))

    # Ровно 250 кредитов
    matrix[0, :] = 1

    # Объём пула: отклонение не более 2%
    matrix[1, :] = data["outstanding_balance"].to_numpy()

    # Доля каждого региона не более 30% от объёма пула
    regions = data["region"].unique()

    for i, region in enumerate(regions[:5]):
        region_mask = (
            data["region"].to_numpy() == region
        )

        matrix[2 + i, :] = (
            data["outstanding_balance"].to_numpy()
            * region_mask
        )

    # Доля кредитов с LTV выше 80% не более 20%
    matrix[7, :] = (
        data["ltv"].to_numpy() > 0.80
    ).astype(float)

    lower = np.array([
        pool_size,
        target_balance * 0.98,
        0, 0, 0, 0, 0, 0
    ])

    upper = np.array([
    pool_size,
    target_balance * 1.02,
    target_balance * 1.02 * 0.30,
    target_balance * 1.02 * 0.30,
    target_balance * 1.02 * 0.30,
    target_balance * 1.02 * 0.30,
    target_balance * 1.02 * 0.30,
    pool_size * 0.20
])

    # Средняя ставка не ниже 15,5%
    rate_constraint = (
        data["interest_rate"].to_numpy() - 0.155
    )

    constraints = [
        LinearConstraint(matrix.tocsr(), lower, upper),
        LinearConstraint(
            rate_constraint.reshape(1, -1),
            0,
            np.inf
        )
    ]

    result = milp(
        c=objective,
        integrality=np.ones(n),
        bounds=Bounds(np.zeros(n), np.ones(n)),
        constraints=constraints,
        options={"time_limit": 60}
    )

    if not result.success:
        raise ValueError(
            f"Оптимизация не выполнена: {result.message}"
        )

    selected = result.x > 0.5

    return data.loc[selected].copy()

st.set_page_config(
    page_title="ПУЛЬСАР | Анализ ипотечных пулов",
    page_icon="🌌",
    layout="wide"
)

st.markdown("""
<style>
    .stApp {
        background-color: #F5F7FA;
        color: #172B4D;
    }

    [data-testid="stSidebar"] {
        background-color: #172B4D;
    }

    [data-testid="stSidebar"] * {
        color: white !important;
    }

    .info-card {
        background: white;
        padding: 22px;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        min-height: 130px;
    }

    .card-label {
        color: #64748B;
        font-size: 14px;
    }

    .card-value {
        color: #172B4D;
        font-size: 25px;
        font-weight: 700;
        margin-top: 12px;
    }

    .section-title {
        font-size: 23px;
        font-weight: 700;
        color: #172B4D;
        margin-top: 25px;
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("# 🌌 ПУЛЬСАР")
    st.caption("АНАЛИТИЧЕСКАЯ ПЛАТФОРМА")
    st.divider()

    section = st.radio(
        "РАЗДЕЛЫ",
        [
            "Главная",
            "Анализ ипотечного пула",
            "Стресс-тестирование",
            "Оптимизация",
            "Отчёт"
        ]
    )

    st.divider()
    st.caption("Версия 1.0 | 2026")


if section == "Главная":

    st.title("ПУЛЬСАР")
    st.subheader("Интеллектуальная система анализа ипотечных пулов")

    st.divider()

    st.markdown("### Аналитическая панель")

    st.write(
        "Комплексный анализ ипотечных портфелей, "
        "моделирование стрессовых сценариев и поиск "
        "оптимального состава ипотечных пулов."
    )

    col1, col2, col3, col4 = st.columns(4)

    metrics = [
    (col1, "Кредитов в пуле", f"{len(df):,}".replace(",", " ")),
    (col2, "Объём портфеля", f"{df['outstanding_balance'].sum() / 1_000_000_000:.2f} млрд ₽"),
    (col3, "Средний LTV", f"{df['ltv'].mean():.1%}"),
    (col4, "Ожидаемые потери", f"{(df['ead'] * df['pd'] * df['lgd']).sum() / 1_000_000:.1f} млн ₽")
]

    for col, label, value in metrics:
        with col:
            st.markdown(
                f"""
                <div class="info-card">
                    <div class="card-label">{label}</div>
                    <div class="card-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown(
        '<div class="section-title">Возможности системы</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="info-card">
            <h4>01 · Анализ портфеля</h4>
            <p>Оценка структуры, качества и концентрации ипотечных кредитов.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-card">
            <h4>02 · Стресс-тестирование</h4>
            <p>Оценка устойчивости пула при изменении экономических условий.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-card">
            <h4>03 · Оптимизация</h4>
            <p>Подбор состава пула с учётом заданных ограничений.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-card">
            <h4>04 · Сравнение</h4>
            <p>Сопоставление альтернативных вариантов формирования пула.</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.caption(
        "ПУЛЬСАР — инструмент поддержки принятия финансовых решений."
    )

elif section == "Анализ ипотечного пула":

    st.title("Анализ ипотечного пула")
    st.write("Структура, качество и риск-профиль ипотечного портфеля.")

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Количество кредитов", f"{len(df):,}".replace(",", " "))

    with col2:
        st.metric("Средний LTV", f"{df['ltv'].mean():.1%}")

    with col3:
        st.metric("Средняя ставка", f"{df['interest_rate'].mean():.1%}")

    st.markdown("### Географическая структура портфеля")

    region_data = df.groupby("region")["outstanding_balance"].sum()

    st.bar_chart(region_data)

    st.markdown("### Распределение кредитов по LTV")

    st.histogram if False else None

    ltv_data = df["ltv"] * 100

    st.bar_chart(
        pd.cut(
            ltv_data,
            bins=[0, 50, 60, 70, 80, 90, 100],
            labels=["до 50%", "50–60%", "60–70%", "70–80%", "80–90%", "90–100%"]
        ).value_counts().sort_index()
    )

    st.markdown("### Данные по ипотечным кредитам")

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

elif section == "Стресс-тестирование":

    st.title("Стресс-тестирование")
    st.write(
        "Оценка устойчивости ипотечного портфеля "
        "при неблагоприятном изменении экономических условий."
    )

    st.divider()

    st.markdown("### Параметры стресс-сценария")

    col1, col2 = st.columns(2)

    with col1:
        collateral_drop = st.slider(
            "Снижение стоимости недвижимости, %",
            min_value=0,
            max_value=40,
            value=20,
            step=5
        )

    with col2:
        pd_multiplier = st.slider(
            "Рост вероятности дефолта (PD), раз",
            min_value=1.0,
            max_value=5.0,
            value=2.0,
            step=0.5
        )

    # Базовые показатели
    base_pd = df["pd"]
    base_lgd = df["lgd"]
    ead = df["ead"]

    base_losses = (ead * base_pd * base_lgd).sum()

    # Стрессовые параметры
    stressed_pd = (base_pd * pd_multiplier).clip(upper=1.0)

    stressed_property = (
        df["property_value"] * (1 - collateral_drop / 100)
    )

    # Учитываем дисконт реализации залога и дополнительные расходы
    recovery = stressed_property * 0.85 * 0.90

    stressed_lgd = (
        1 - recovery / ead
    ).clip(lower=0, upper=1)

    stressed_losses = (
        ead * stressed_pd * stressed_lgd
    ).sum()

    additional_losses = stressed_losses - base_losses

    st.divider()

    st.markdown("### Результаты стресс-тестирования")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Базовые ожидаемые потери",
            f"{base_losses / 1_000_000:.1f} млн ₽"
        )

    with col2:
        st.metric(
            "Потери при стрессе",
            f"{stressed_losses / 1_000_000:.1f} млн ₽",
            delta=f"{additional_losses / 1_000_000:+.1f} млн ₽",
            delta_color="inverse"
        )

    with col3:
        increase = (
            (stressed_losses / base_losses - 1) * 100
            if base_losses > 0 else 0
        )

        st.metric(
            "Рост потерь",
            f"{increase:.1f}%"
        )

    st.markdown("### Сравнение сценариев")

    comparison = pd.DataFrame({
        "Сценарий": ["Базовый", "Стрессовый"],
        "Ожидаемые потери, млн ₽": [
            base_losses / 1_000_000,
            stressed_losses / 1_000_000
        ]
    }).set_index("Сценарий")

    st.bar_chart(comparison)

    st.caption(
        "Расчёт основан на синтетических данных. "
        "Стрессовый LGD учитывает снижение стоимости залога "
        "и условные расходы на его реализацию в размере 10%."
    )

elif section == "Оптимизация":

    st.title("Оптимизация ипотечного пула")
    st.write(
        "Подбор состава ипотечного пула с минимальными "
        "ожидаемыми потерями при заданных ограничениях."
    )

    st.divider()

    st.markdown("### Параметры оптимизации")

    st.write("Алгоритм будет подбирать пул со следующими условиями:")

    st.markdown("""
    - Количество кредитов: **250**
    - Целевой объём: **1,25 млрд ₽** (допуск ±2%)
    - Доля одного региона: **не более 30%**
    - Доля кредитов с LTV выше 80%: **не более 20%**
    - Средняя процентная ставка: **не ниже 15,5%**
    """)

    st.divider()
    st.markdown("### Параметры стресс-сценария")

    col1, col2 = st.columns(2)

    with col1:
        collateral_drop = st.slider(
            "Снижение стоимости недвижимости, %",
            min_value=0,
            max_value=40,
            value=20,
            step=5,
            key="optimization_collateral_drop"
        )

    with col2:
        pd_multiplier = st.slider(
            "Рост вероятности дефолта (PD), раз",
            min_value=1.0,
            max_value=5.0,
            value=2.0,
            step=0.5,
            key="optimization_pd_multiplier"
        )

    collateral_drop = collateral_drop / 100
    if st.button("Запустить оптимизацию", type="primary"):

        with st.spinner("Подбираем оптимальный состав пула..."):

            try:
                random_pool = optimize_pool(
                    df, mode="random",
                    collateral_drop=collateral_drop,
                    pd_multiplier=pd_multiplier
                )

                quality_pool = optimize_pool(
                    df, mode="quality",
                    collateral_drop=collateral_drop,
                    pd_multiplier=pd_multiplier
                )

                optimized = optimize_pool(
                    df, mode="stress",
                    collateral_drop=collateral_drop,
                    pd_multiplier=pd_multiplier
                )
                st.session_state["random_pool"] = random_pool
                st.session_state["quality_pool"] = quality_pool
                st.session_state["optimized"] = optimized

                st.session_state["optimization_params"] = {
                    "collateral_drop": collateral_drop,
                    "pd_multiplier": pd_multiplier
                }

                st.success("Оптимальный состав пула найден!")
                def calculate_stress_losses(pool):
                    stress_pd = np.clip(
                        pool["pd"].to_numpy() * pd_multiplier,
                        0,
                        1
                    )

                    stressed_property = (
                        pool["property_value"].to_numpy()
                        * (1 - collateral_drop)
                    )

                    recovery = stressed_property * 0.85 * 0.9

                    stressed_lgd = np.clip(
                        1 - recovery / pool["ead"].to_numpy(),
                        0,
                        1
                    )

                    return (
                        pool["ead"].to_numpy()
                        * stress_pd
                        * stressed_lgd
                    ).sum()

                comparison = pd.DataFrame({
                    "Вариант": [
                        "Случайный отбор",
                        "По текущему качеству",
                        "ПУЛЬСАР"
                    ],
                    "Количество кредитов": [
                        len(random_pool),
                        len(quality_pool),
                        len(optimized)
                    ],
                    "Объём, млн ₽": [
                        random_pool["ead"].sum() / 1_000_000,
                        quality_pool["ead"].sum() / 1_000_000,
                        optimized["ead"].sum() / 1_000_000
                    ],
                    "Средний LTV, %": [
                        random_pool["ltv"].mean() * 100,
                        quality_pool["ltv"].mean() * 100,
                        optimized["ltv"].mean() * 100
                    ],
                    "Кредиты с LTV выше 80%, %": [
                        (random_pool["ltv"] > 0.80).mean() * 100,
                        (quality_pool["ltv"] > 0.80).mean() * 100,
                        (optimized["ltv"] > 0.80).mean() * 100
                    ],
                    "Максимальная доля региона, %": [
                        random_pool.groupby("region")["ead"].sum().max()
                        / random_pool["ead"].sum() * 100,
                        quality_pool.groupby("region")["ead"].sum().max()
                        / quality_pool["ead"].sum() * 100,
                        optimized.groupby("region")["ead"].sum().max()
                        / optimized["ead"].sum() * 100
                    ],
                    "Средняя ставка, %": [
                        random_pool["interest_rate"].mean() * 100,
                        quality_pool["interest_rate"].mean() * 100,
                        optimized["interest_rate"].mean() * 100
                    ],
                    "Стрессовые потери, млн ₽": [
                        calculate_stress_losses(random_pool) / 1_000_000,
                        calculate_stress_losses(quality_pool) / 1_000_000,
                        calculate_stress_losses(optimized) / 1_000_000
                    ],
                    "Стрессовые потери, % от EAD": [
                        calculate_stress_losses(random_pool)
                        / random_pool["ead"].sum() * 100,
                        calculate_stress_losses(quality_pool)
                        / quality_pool["ead"].sum() * 100,
                        calculate_stress_losses(optimized)
                        / optimized["ead"].sum() * 100
                    ]
                })
                st.session_state["comparison"] = comparison.copy()
                st.session_state["random_pool"] = random_pool.copy()
                st.session_state["quality_pool"] = quality_pool.copy()
                st.session_state["optimized"] = optimized.copy()

                st.markdown("### Сравнение вариантов формирования пула")

                st.dataframe(
                    comparison.round(4),
                    use_container_width=True,
                    hide_index=True
                )

                # Основные результаты оптимизации
                total_balance = optimized["outstanding_balance"].sum()

                expected_losses = (
                    optimized["ead"]
                    * optimized["pd"]
                    * optimized["lgd"]
                ).sum()

                stress_losses = calculate_stress_losses(optimized)

                st.markdown("### Результаты оптимизации")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Количество кредитов",
                        f"{len(optimized):,}"
                    )

                with col2:
                    st.metric(
                        "Объём пула",
                        f"{total_balance / 1_000_000:,.2f} млн ₽"
                    )

                with col3:
                    st.metric(
                        "Средний LTV",
                        f"{optimized['ltv'].mean():.1%}"
                    )

                col4, col5, col6 = st.columns(3)

                with col4:
                    st.metric(
                        "Кредиты с LTV выше 80%",
                        f"{(optimized['ltv'] > 0.80).mean():.1%}"
                    )

                with col5:
                    st.metric(
                        "Базовые ожидаемые потери",
                        f"{expected_losses / 1_000_000:.2f} млн ₽"
                    )

                with col6:
                    st.metric(
                        "Стрессовые потери",
                        f"{stress_losses / 1_000_000:.2f} млн ₽"
                    )
                # ==================================
                # СРАВНИТЕЛЬНЫЕ ДИАГРАММЫ ПУЛОВ
                # ==================================

                st.divider()
                st.markdown("## Визуальное сравнение пулов")

                # 1. Сравнение распределения LTV

                st.markdown("### 1. Распределение кредитов по LTV")

                ltv_comparison = []

                for pool, name in [
                    (random_pool, "Случайный отбор"),
                    (quality_pool, "По текущему качеству"),
                    (optimized, "ПУЛЬСАР")
                ]:
                    ltv_values = pool["ltv"].to_numpy() * 100

                    counts, bin_edges = np.histogram(
                        ltv_values,
                        bins=np.linspace(40, 100, 26)
                    )

                    bin_centers = (
                        bin_edges[:-1] + bin_edges[1:]
                    ) / 2

                    shares = counts / len(pool) * 100

                    for center, share in zip(bin_centers, shares):
                        ltv_comparison.append({
                            "LTV, %": center,
                            "Доля кредитов, %": share,
                            "Вариант": name
                        })

                ltv_comparison = pd.DataFrame(ltv_comparison)

                fig_ltv = px.line(
                    ltv_comparison,
                    x="LTV, %",
                    y="Доля кредитов, %",
                    color="Вариант",
                    markers=True,
                    color_discrete_map={
                        "Случайный отбор": "#9FA764",
                        "По текущему качеству": "#816A60",
                        "ПУЛЬСАР": "#613C4E"
                    }
                )

                fig_ltv.update_traces(
                    line=dict(width=3),
                    marker=dict(size=6)
                )

                fig_ltv.update_layout(
                    xaxis_title="LTV, %",
                    yaxis_title="Доля кредитов, %",
                    legend_title="Вариант формирования пула",
                    xaxis=dict(range=[40, 100]),
                    height=450,
                    hovermode="x unified"
                )

                st.plotly_chart(
                    fig_ltv,
                    use_container_width=True
                )

                st.caption(
                    "Каждая линия показывает долю кредитов "
                    "в соответствующем диапазоне LTV. "
                    "Чем выше линия, тем больше кредитов "
                    "сосредоточено в этом диапазоне."
                )

                # 2. Сравнение стрессовых потерь

                st.markdown("### 2. Сравнение стрессовых потерь")

                fig_losses = px.bar(
                    comparison,
                    x="Вариант",
                    y="Стрессовые потери, % от EAD",
                    text="Стрессовые потери, % от EAD",
                    color="Вариант",
                    color_discrete_map={
                        "Случайный отбор": "#9FA764",
                        "По текущему качеству": "#816A60",
                        "ПУЛЬСАР": "#613C4E"
                    },
                    labels={
                        "Вариант": "Способ формирования пула",
                        "Стрессовые потери, % от EAD":
                            "Потери, % от EAD"
                    }
                )

                fig_losses.update_traces(
                    texttemplate="<b>%{text:.4f}%</b>",
                    textposition="outside",
                    textfont=dict(size=16, color="#302B1A"),
                    cliponaxis=False
                )

                fig_losses.update_layout(
                    yaxis=dict(
                        type="linear",
                        title="Стрессовые потери, % от EAD",
                        range=[0, 0.65],
                        dtick=0.10,
                        tickformat=".2f",
                        ticksuffix="%"
                    ),
                    xaxis_title="",
                    showlegend=False,
                    height=480,
                    margin=dict(t=70, b=80),
                    font=dict(size=14, color="#302B1A"),
                    xaxis=dict(
                        tickfont=dict(size=14, color="#302B1A")
                    )
                )

                st.plotly_chart(
                    fig_losses,
                    use_container_width=True
                )

                st.caption(
                    "Логарифмическая шкала позволяет сравнить "
                    "даже существенно различающиеся значения потерь. "
                    "Чем ниже столбец, тем меньше стрессовые потери."
                )

                # 3. Региональная структура пулов

                st.markdown("### 3. Сравнение региональной структуры пулов")

                region_comparison = []

                for pool, name in [
                    (random_pool, "Случайный отбор"),
                    (quality_pool, "По текущему качеству"),
                    (optimized, "ПУЛЬСАР")
                ]:
                    regional_balance = (
                        pool.groupby("region")["ead"].sum()
                    )

                    regional_share = (
                        regional_balance / regional_balance.sum() * 100
                    )

                    for region, share in regional_share.items():
                        region_comparison.append({
                            "Вариант": name,
                            "Регион": region,
                            "Доля объёма, %": share
                        })

                region_comparison = pd.DataFrame(region_comparison)

                fig_regions = px.bar(
                    region_comparison,
                    x="Регион",
                    y="Доля объёма, %",
                    color="Вариант",
                    barmode="group",
                    text="Доля объёма, %",
                    color_discrete_map={
                        "Случайный отбор": "#9FA764",
                        "По текущему качеству": "#816A60",
                        "ПУЛЬСАР": "#613C4E"
                    },
                    labels={
                        "Регион": "Регион",
                        "Доля объёма, %": "Доля объёма пула, %",
                        "Вариант": "Способ формирования пула"
                    }
                )

                fig_regions.update_traces(
                    texttemplate="%{text:.1f}%",
                    textposition="outside",
                    textfont=dict(size=11),
                    cliponaxis=False
                )

                fig_regions.update_layout(
                    barmode="group",
                    yaxis_title="Доля объёма пула, %",
                    xaxis_title="",
                    height=500,
                    margin=dict(t=40, b=100),
                    font=dict(size=13, color="#302B1A"),
                    legend_title="Вариант",
                    yaxis=dict(
                        rangemode="tozero"
                    )
                )

                st.plotly_chart(
                    fig_regions,
                    use_container_width=True
                )

                st.caption(
                    "Сравнение долей пяти регионов в общем объёме "
                    "трёх ипотечных пулов. Позволяет оценить "
                    "региональную концентрацию и различия "
                    "между способами формирования пулов."
                )

                # Формирование Excel-файла
                excel_buffer = BytesIO()

                with pd.ExcelWriter(
                    excel_buffer,
                    engine="openpyxl"
                ) as writer:

                    optimized.to_excel(
                        writer,
                        sheet_name="Оптимизированный пул",
                        index=False
                    )

                    comparison.to_excel(
                        writer,
                        sheet_name="Сравнение вариантов",
                        index=False
                    )

                    # Оформление листов
                    for sheet in writer.sheets.values():

                        sheet.freeze_panes = "A2"
                        sheet.auto_filter.ref = sheet.dimensions

                        for cell in sheet[1]:
                            cell.font = Font(
                                bold=True,
                                color="FFFFFF"
                            )
                            cell.fill = PatternFill(
                                "solid",
                                fgColor="304B63"
                            )
                            cell.alignment = Alignment(
                                horizontal="center",
                                vertical="center",
                                wrap_text=True
                            )

                        sheet.row_dimensions[1].height = 35

                        for column in sheet.columns:
                            max_length = max(
                                len(str(cell.value or ""))
                                for cell in column
                            )

                            column_letter = get_column_letter(
                                column[0].column
                            )

                            sheet.column_dimensions[
                                column_letter
                            ].width = min(max_length + 3, 30)

                st.download_button(
                    "Скачать Excel-файл ПУЛЬСАР",
                    data=excel_buffer.getvalue(),
                    file_name="PULSAR_результаты.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

            except Exception as e:
                st.error(f"Не удалось выполнить оптимизацию: {e}")


elif section == "Отчёт":

    st.title("Аналитический отчёт")
    st.write(
        "Формирование итогового отчёта по результатам "
        "анализа, стресс-тестирования и оптимизации "
        "ипотечного пула."
    )

    st.divider()

    st.subheader("Формирование отчёта")

    st.write(
        "Отчёт будет содержать основные характеристики "
        "ипотечного пула, результаты стресс-тестирования, "
        "сравнение вариантов формирования пула и "
        "итоги оптимизации."
    )

    if "comparison" not in st.session_state:
        st.warning(
            "Сначала выполните оптимизацию ипотечного пула, "
            "чтобы сформировать отчёт."
        )
    else:
        st.success("Данные для формирования отчёта доступны.")
        pdf_file = create_pdf_report(
            st.session_state["comparison"],
            st.session_state["optimization_params"]
        )

        st.download_button(
            label="Скачать PDF-отчёт",
            data=pdf_file,
            file_name="ПУЛЬСАР_аналитический_отчёт.pdf",
            mime="application/pdf",
            use_container_width=True
        )