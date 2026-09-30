import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Customer Segmentation & RFM Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container {
    padding-top: 1.3rem;
    padding-bottom: 2rem;
}

[data-testid="stMetric"] {
    border: 1px solid rgba(128,128,128,.18);
    padding: 12px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# COLUMN ALIASES
# ============================================================

ALIASES = {
    "invoice": [
        "invoiceno", "invoice", "orderid", "order_id",
        "transactionid", "transaction_id"
    ],
    "product": [
        "stockcode", "productid", "product_id",
        "sku", "itemid", "item_id"
    ],
    "description": [
        "description", "product", "productname",
        "product_name", "item"
    ],
    "quantity": [
        "quantity", "qty", "units"
    ],
    "date": [
        "invoicedate", "invoice_date", "orderdate",
        "order_date", "date", "transactiondate",
        "transaction_date"
    ],
    "price": [
        "unitprice", "unit_price", "price",
        "amount", "sales"
    ],
    "customer": [
        "customerid", "customer_id", "customer",
        "userid", "user_id"
    ],
    "country": [
        "country", "region", "market", "location"
    ],
}


# ============================================================
# DATA FUNCTIONS
# ============================================================

def normalize_columns(df):
    df = df.copy()

    df.columns = [
        str(c)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        for c in df.columns
    ]

    return df


def find_column(columns, aliases):
    for alias in aliases:
        if alias in columns:
            return alias

    return None


def standardize_input(raw):
    df = normalize_columns(raw)

    mapping = {}

    for target, aliases in ALIASES.items():
        found = find_column(df.columns, aliases)

        if found:
            mapping[target] = found

    required = [
        "invoice",
        "quantity",
        "date",
        "price",
        "customer"
    ]

    missing = [
        x for x in required
        if x not in mapping
    ]

    if missing:
        raise ValueError(
            "Required columns could not be detected: "
            + ", ".join(missing)
        )

    rename_map = {
        source: target
        for target, source in mapping.items()
    }

    df = df.rename(columns=rename_map)

    if "product" not in df.columns:
        df["product"] = "Unknown"

    if "description" not in df.columns:
        df["description"] = df["product"].astype(str)

    if "country" not in df.columns:
        df["country"] = "Unknown"

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df["quantity"] = pd.to_numeric(
        df["quantity"],
        errors="coerce"
    )

    df["price"] = pd.to_numeric(
        df["price"],
        errors="coerce"
    )

    df["customer"] = (
        df["customer"]
        .astype(str)
        .str.strip()
    )

    df["invoice"] = (
        df["invoice"]
        .astype(str)
        .str.strip()
    )

    df["revenue"] = (
        df["quantity"] *
        df["price"]
    )

    clean = df.dropna(
        subset=[
            "date",
            "quantity",
            "price",
            "customer",
            "invoice"
        ]
    ).copy()

    clean = clean[
        (clean["customer"] != "")
        & (clean["customer"] != "nan")
        & (clean["quantity"] > 0)
        & (clean["price"] > 0)
        & (clean["revenue"] > 0)
    ]

    return clean, mapping


# ============================================================
# RFM SCORING
# ============================================================

def score_metric(series, ascending=True):

    ranked = series.rank(
        method="first",
        ascending=ascending
    )

    return pd.qcut(
        ranked,
        5,
        labels=[1, 2, 3, 4, 5]
    ).astype(int)


def build_rfm(df):

    analysis_date = (
        df["date"].max().normalize()
        + pd.Timedelta(days=1)
    )

    rfm = (
        df.groupby("customer")
        .agg(
            LastPurchase=("date", "max"),
            Frequency=("invoice", "nunique"),
            Monetary=("revenue", "sum"),
            TotalUnits=("quantity", "sum"),
            UniqueProducts=("product", "nunique"),
            Country=("country", "first"),
        )
        .reset_index()
    )

    rfm["Recency"] = (
        analysis_date -
        rfm["LastPurchase"].dt.normalize()
    ).dt.days

    rfm["Monetary"] = (
        rfm["Monetary"]
        .round(2)
    )

    # Lower recency is better.
    rfm["R_Score"] = score_metric(
        rfm["Recency"],
        ascending=False
    )

    rfm["F_Score"] = score_metric(
        rfm["Frequency"],
        ascending=True
    )

    rfm["M_Score"] = score_metric(
        rfm["Monetary"],
        ascending=True
    )

    rfm["RFM_Score"] = (
        rfm["R_Score"].astype(str)
        + rfm["F_Score"].astype(str)
        + rfm["M_Score"].astype(str)
    )

    rfm["RFM_Total"] = (
        rfm["R_Score"]
        + rfm["F_Score"]
        + rfm["M_Score"]
    )

    rfm["AOV"] = (
        rfm["Monetary"] /
        rfm["Frequency"]
    ).round(2)

    # ========================================================
    # PURCHASE GAP ANALYSIS
    # ========================================================

    purchases = (
        df[
            [
                "customer",
                "invoice",
                "date"
            ]
        ]
        .drop_duplicates()
        .sort_values(
            ["customer", "date"]
        )
    )

    purchases["gap"] = (
        purchases
        .groupby("customer")["date"]
        .diff()
        .dt.days
    )

    gap_stats = (
        purchases
        .groupby("customer")["gap"]
        .agg(
            AvgOrderGap="mean",
            MedianOrderGap="median"
        )
        .reset_index()
    )

    rfm = rfm.merge(
        gap_stats,
        on="customer",
        how="left"
    )

    rfm["ExpectedInterval"] = (
        rfm["MedianOrderGap"]
        .fillna(60)
        .clip(14, 180)
    )

    rfm["PurchasePressure"] = (
        rfm["Recency"] /
        rfm["ExpectedInterval"]
    ).clip(0, 5)

    # ========================================================
    # EXPLAINABLE RISK SCORE
    # ========================================================

    recency_pressure = (
        rfm["PurchasePressure"] / 2.5
    ).clip(0, 1)

    low_frequency = (
        1 -
        rfm["F_Score"] / 5
    ).clip(0, 1)

    low_value = (
        1 -
        rfm["M_Score"] / 5
    ).clip(0, 1)

    rfm["RiskScore"] = (
        0.65 * recency_pressure
        + 0.20 * low_frequency
        + 0.15 * low_value
    ).clip(0, 1)

    rfm["RiskBand"] = pd.cut(
        rfm["RiskScore"],
        bins=[
            -0.01,
            0.33,
            0.66,
            1.01
        ],
        labels=[
            "Low",
            "Medium",
            "High"
        ]
    )

    # ========================================================
    # CUSTOMER SEGMENTATION
    # ========================================================

    def segment(row):

        r = row["R_Score"]
        f = row["F_Score"]
        m = row["M_Score"]

        if r >= 4 and f >= 4 and m >= 4:
            return "Champions"

        if r >= 3 and f >= 4 and m >= 3:
            return "Loyal"

        if r >= 4 and f <= 2:
            return "New / Promising"

        if r >= 3 and m >= 4:
            return "Big Spenders"

        if r <= 2 and f >= 3:
            return "At Risk"

        if r <= 2 and f <= 2:
            return "Lost"

        return "Regular"

    rfm["Segment"] = rfm.apply(
        segment,
        axis=1
    )

    actions = {

        "Champions":
            "VIP treatment, early access and referrals",

        "Loyal":
            "Loyalty rewards and cross-selling",

        "New / Promising":
            "Second-purchase incentive",

        "Big Spenders":
            "Premium personalized offers",

        "At Risk":
            "Win-back campaign",

        "Lost":
            "Low-cost reactivation test",

        "Regular":
            "Frequency-building offers",
    }

    rfm["RecommendedAction"] = (
        rfm["Segment"]
        .map(actions)
    )

    # ========================================================
    # CLV PROXY
    # ========================================================

    active_days = max(
        1,
        (
            df["date"].max()
            - df["date"].min()
        ).days
    )

    years = max(
        active_days / 365.25,
        0.5
    )

    annual_order_rate = (
        rfm["Frequency"] /
        years
    )

    rfm["AnnualValue"] = (
        rfm["AOV"] *
        annual_order_rate
    )

    rfm["CLV_Proxy"] = (
        rfm["AnnualValue"] *
        1.5
    ).round(2)

    return (
        rfm,
        analysis_date,
        years
    )


# ============================================================
# GRAPH BORDER
# ============================================================

def add_graph_border(fig):

    fig.update_layout(
        shapes=[
            dict(
                type="rect",
                xref="paper",
                yref="paper",

                x0=0,
                y0=0,

                x1=1,
                y1=1,

                line=dict(
                    color="#3A3A3A",
                    width=1
                ),

                fillcolor="rgba(0,0,0,0)"
            )
        ]
    )

    return fig


# ============================================================
# RETENTION PLANNER
# ============================================================

def create_retention_plan(rfm):

    plan = (
        rfm
        .groupby("Segment")
        .agg(
            Customers=("customer", "nunique"),
            Revenue=("Monetary", "sum"),
            CLV=("CLV_Proxy", "sum"),
            HighRiskCustomers=(
                "RiskBand",
                lambda x:
                int((x == "High").sum())
            )
        )
        .reset_index()
    )

    total_revenue = max(
        plan["Revenue"].sum(),
        1
    )

    total_clv = max(
        plan["CLV"].sum(),
        1
    )

    plan["RevenueShare"] = (
        plan["Revenue"] /
        total_revenue
    )

    plan["CLVShare"] = (
        plan["CLV"] /
        total_clv
    )

    max_risk = max(
        plan["HighRiskCustomers"].max(),
        1
    )

    plan["PriorityIndex"] = (
        0.55 * plan["CLVShare"]
        + 0.30 * (
            plan["HighRiskCustomers"] /
            max_risk
        )
        + 0.15 * plan["RevenueShare"]
    )

    eligible = plan["Segment"].isin([
        "Champions",
        "Loyal",
        "Big Spenders",
        "At Risk",
        "New / Promising"
    ])

    priority = (
        plan["PriorityIndex"]
        .where(eligible, 0)
    )

    if priority.sum() == 0:
        priority = plan["RevenueShare"]

    plan["SuggestedBudgetPct"] = (
        priority /
        priority.sum() *
        100
    ).round(1)

    plan["SuggestedAction"] = (
        plan["Segment"].map({

            "Champions":
                "VIP / referral / early access",

            "Loyal":
                "Loyalty reward + cross-sell",

            "New / Promising":
                "Second-order incentive",

            "Big Spenders":
                "Premium personalized offer",

            "At Risk":
                "Win-back campaign",

            "Lost":
                "Cheap reactivation test",

            "Regular":
                "Frequency-building promotion",
        })
    )

    return plan.sort_values(
        "SuggestedBudgetPct",
        ascending=False
    )


# ============================================================
# LOAD DATA
# ============================================================

st.sidebar.title("📊 RFM Analytics")

st.sidebar.caption(
    "Customer segmentation + retention planning"
)

uploaded = st.sidebar.file_uploader(
    "Upload transaction CSV",
    type=["csv"]
)

sample_path = (
    Path(__file__).parent
    / "data"
    / "sample_transactions.csv"
)

if uploaded is not None:

    raw_df = pd.read_csv(
        uploaded
    )

    source_name = uploaded.name

else:

    if not sample_path.exists():

        st.error(
            "Demo dataset not found. "
            "Make sure data/sample_transactions.csv exists."
        )

        st.stop()

    raw_df = pd.read_csv(
        sample_path
    )

    source_name = "Built-in demo dataset"


# ============================================================
# CLEAN DATA
# ============================================================

try:

    df, mapping = standardize_input(
        raw_df
    )

except Exception as error:

    st.error(
        str(error)
    )

    st.stop()


if df.empty:

    st.error(
        "No valid transaction data remains."
    )

    st.stop()


# ============================================================
# BUILD RFM
# ============================================================

rfm, analysis_date, years = build_rfm(
    df
)


# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.markdown("---")

countries = sorted(
    rfm["Country"]
    .astype(str)
    .unique()
)

selected_countries = st.sidebar.multiselect(
    "Country",
    countries,
    default=countries
)

segments = sorted(
    rfm["Segment"]
    .unique()
)

selected_segments = st.sidebar.multiselect(
    "Segment",
    segments,
    default=segments
)

selected_risk = st.sidebar.multiselect(
    "Risk Band",
    ["Low", "Medium", "High"],
    default=[
        "Low",
        "Medium",
        "High"
    ]
)

filtered = rfm[
    rfm["Country"]
    .astype(str)
    .isin(selected_countries)
    &
    rfm["Segment"]
    .isin(selected_segments)
    &
    rfm["RiskBand"]
    .astype(str)
    .isin(selected_risk)
].copy()

st.sidebar.markdown("---")

st.sidebar.caption(
    f"Source: {source_name}"
)



# ============================================================
# HEADER
# ============================================================

st.title(
    "Customer Segmentation & RFM Analysis"
)

st.markdown(
    """
    End-to-end customer analytics dashboard using
    **RFM scoring, customer segmentation, risk analysis,
    CLV estimation and retention planning**.
    """
)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📌 Executive Dashboard",
    "🎯 RFM Explorer",
    "👤 Customer 360",
    "💰 Retention Planner",
    "🧪 Data Quality & Logic"
])


# ============================================================
# TAB 1
# ============================================================

with tab1:

    total_revenue = (
        filtered["Monetary"].sum()
    )

    customer_count = len(
        filtered
    )

    average_aov = (
        filtered["AOV"].mean()
        if customer_count
        else 0
    )

    high_risk = int(
        (
            filtered["RiskBand"]
            == "High"
        ).sum()
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Customers",
        f"{customer_count:,}"
    )

    c2.metric(
        "Revenue",
        f"£{total_revenue:,.0f}"
    )

    c3.metric(
        "Average AOV",
        f"£{average_aov:,.2f}"
    )

    c4.metric(
        "High-Risk Customers",
        f"{high_risk:,}"
    )

    st.markdown(
        "### Segment Performance"
    )

    segment_data = (
        filtered
        .groupby("Segment")
        .agg(
            Customers=("customer", "nunique"),
            Revenue=("Monetary", "sum"),
            CLV=("CLV_Proxy", "sum")
        )
        .reset_index()
    )

    col1, col2 = st.columns(2)

    with col1:

        fig = px.bar(
            segment_data.sort_values(
                "Revenue",
                ascending=False
            ),
            x="Segment",
            y="Revenue",
            text_auto=".2s",
            title="Revenue by Customer Segment"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig = px.pie(
            segment_data,
            names="Segment",
            values="Customers",
            hole=0.45,
            title="Customer Segment Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.markdown(
        "### Revenue Concentration"
    )

    if not filtered.empty:

        pareto = (
            filtered
            .sort_values(
                "Monetary",
                ascending=False
            )
            .reset_index(drop=True)
        )

        pareto["CumulativeRevenue"] = (
            pareto["Monetary"].cumsum()
            /
            pareto["Monetary"].sum()
            *
            100
        )

        pareto["CustomerPct"] = (
            (np.arange(len(pareto)) + 1)
            /
            len(pareto)
            *
            100
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=pareto["CustomerPct"],
                y=pareto["CumulativeRevenue"],
                mode="lines",
                name="Cumulative Revenue"
            )
        )

        fig.add_shape(
            type="line",
            x0=0,
            x1=100,
            y0=80,
            y1=80,
            line=dict(
                dash="dash"
            )
        )

        fig.update_layout(
            title="Revenue Concentration / Pareto Analysis",
            xaxis_title="% of Customers",
            yaxis_title="Cumulative Revenue %",
            yaxis_range=[0, 105]
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# TAB 2 — RFM EXPLORER
# ============================================================

with tab2:

    st.markdown(
        "### RFM Distribution"
    )

    c1, c2, c3 = st.columns(3)

    # ========================================================
    # RECENCY
    # ========================================================

    with c1:

        fig = px.histogram(
            filtered,
            x="Recency",
            nbins=30,
            title="Recency distribution",
            color_discrete_sequence=[
                "#7EC3F5"
            ]
        )

        # BORDER AROUND ENTIRE GRAPH
        add_graph_border(fig)

        # BORDER AROUND INDIVIDUAL BARS
        fig.update_traces(
            marker_line_color="#333333",
            marker_line_width=1
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # ========================================================
    # FREQUENCY
    # ========================================================

    with c2:

        fig = px.histogram(
            filtered,
            x="Frequency",
            nbins=20,
            title="Order frequency",
            color_discrete_sequence=[
                "#7EC3F5"
            ]
        )

        # BORDER AROUND ENTIRE GRAPH
        add_graph_border(fig)

        # BORDER AROUND INDIVIDUAL BARS
        fig.update_traces(
            marker_line_color="#333333",
            marker_line_width=1
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # ========================================================
    # MONETARY
    # ========================================================

    with c3:

        fig = px.histogram(
            filtered,
            x="Monetary",
            nbins=30,
            title="Customer monetary value",
            color_discrete_sequence=[
                "#7EC3F5"
            ]
        )

        # BORDER AROUND ENTIRE GRAPH
        add_graph_border(fig)

        # BORDER AROUND INDIVIDUAL BARS
        fig.update_traces(
            marker_line_color="#333333",
            marker_line_width=1
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # ========================================================
    # CUSTOMER VALUE MAP
    # ========================================================

    st.markdown(
        "### Customer Value Map"
    )

    fig = px.scatter(
        filtered,
        x="Recency",
        y="Monetary",
        size="Frequency",
        color="Segment",
        hover_data=[
            "customer",
            "RFM_Score",
            "RiskBand",
            "CLV_Proxy"
        ],
        title="Recency vs Monetary Value",
        size_max=35
    )

    fig.update_xaxes(
        autorange="reversed"
    )

    add_graph_border(fig)

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ========================================================
    # RFM TABLE
    # ========================================================

    st.markdown(
        "### Customer RFM Table"
    )

    display_columns = [
        "customer",
        "Segment",
        "Recency",
        "Frequency",
        "Monetary",
        "RFM_Score",
        "RFM_Total",
        "RiskBand",
        "RiskScore",
        "AOV",
        "CLV_Proxy",
        "RecommendedAction"
    ]

    table = (
        filtered[display_columns]
        .sort_values(
            [
                "RFM_Total",
                "Monetary"
            ],
            ascending=False
        )
    )

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "⬇️ Download Customer RFM Data",
        table.to_csv(
            index=False
        ).encode("utf-8"),
        "customer_rfm_segments.csv",
        "text/csv"
    )


# ============================================================
# TAB 3 — CUSTOMER 360
# ============================================================

with tab3:

    st.markdown(
        "### Customer 360"
    )

    customer_list = (
        filtered
        .sort_values(
            "Monetary",
            ascending=False
        )["customer"]
        .tolist()
    )

    if not customer_list:

        st.warning(
            "No customers match the selected filters."
        )

    else:

        selected_customer = st.selectbox(
            "Select Customer",
            customer_list
        )

        customer = filtered[
            filtered["customer"]
            == selected_customer
        ].iloc[0]

        c1, c2, c3, c4, c5 = st.columns(5)

        c1.metric(
            "Segment",
            customer["Segment"]
        )

        c2.metric(
            "RFM Score",
            customer["RFM_Score"]
        )

        c3.metric(
            "Recency",
            f"{int(customer['Recency'])} days"
        )

        c4.metric(
            "Orders",
            int(customer["Frequency"])
        )

        c5.metric(
            "Total Spend",
            f"£{customer['Monetary']:,.2f}"
        )

        st.markdown(
            "### Customer Analysis"
        )

        explanation = pd.DataFrame({

            "Metric": [
                "Recency",
                "Frequency",
                "Monetary",
                "RFM Score",
                "Risk",
                "AOV",
                "CLV Proxy"
            ],

            "Value": [

                f"{int(customer['Recency'])} days",

                f"{int(customer['Frequency'])} orders",

                f"£{customer['Monetary']:,.2f}",

                customer["RFM_Score"],

                (
                    f"{customer['RiskBand']} "
                    f"({customer['RiskScore']:.2f})"
                ),

                f"£{customer['AOV']:,.2f}",

                f"£{customer['CLV_Proxy']:,.2f}"
            ]
        })

        st.dataframe(
            explanation,
            use_container_width=True,
            hide_index=True
        )

        st.success(
            "Recommended action: "
            + str(
                customer["RecommendedAction"]
            )
        )

        customer_transactions = (
            df[
                df["customer"]
                == selected_customer
            ]
            .sort_values("date")
        )

        if not customer_transactions.empty:

            monthly = (
                customer_transactions
                .assign(
                    Month=
                    customer_transactions["date"]
                    .dt.to_period("M")
                    .astype(str)
                )
                .groupby("Month")
                ["revenue"]
                .sum()
                .reset_index()
            )

            fig = px.bar(
                monthly,
                x="Month",
                y="revenue",
                title="Customer Purchase History"
            )

            add_graph_border(fig)

            st.plotly_chart(
                fig,
                use_container_width=True
            )


# ============================================================
# TAB 4 — RETENTION PLANNER
# ============================================================

with tab4:

    st.markdown(
        "### Retention Campaign Planner"
    )

    st.write(
        """
        The planner creates a transparent budget allocation
        using customer value, revenue contribution and
        high-risk customer concentration.
        """
    )

    plan = create_retention_plan(
        filtered
    )

    total_budget = st.number_input(
        "Campaign Budget",
        min_value=0.0,
        value=10000.0,
        step=500.0
    )

    plan["SuggestedBudget"] = (
        plan["SuggestedBudgetPct"]
        /
        100
        *
        total_budget
    ).round(2)

    st.dataframe(
        plan[
            [
                "Segment",
                "Customers",
                "Revenue",
                "CLV",
                "HighRiskCustomers",
                "RevenueShare",
                "PriorityIndex",
                "SuggestedBudgetPct",
                "SuggestedBudget",
                "SuggestedAction"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )

    col1, col2 = st.columns(2)

    with col1:

        fig = px.bar(
            plan.sort_values(
                "SuggestedBudget"
            ),
            x="SuggestedBudget",
            y="Segment",
            orientation="h",
            title="Suggested Retention Budget"
        )

        add_graph_border(fig)

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig = px.bar(
            plan.sort_values(
                "CLV"
            ),
            x="CLV",
            y="Segment",
            orientation="h",
            title="CLV Proxy by Segment"
        )

        add_graph_border(fig)

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.download_button(
        "⬇️ Download Retention Plan",
        plan.to_csv(
            index=False
        ).encode("utf-8"),
        "retention_campaign_plan.csv",
        "text/csv"
    )


# ============================================================
# TAB 5 — DATA QUALITY
# ============================================================

with tab5:

    st.markdown(
        "### Data Quality Checks"
    )

    raw_rows = len(
        raw_df
    )

    clean_rows = len(
        df
    )

    excluded_rows = (
        raw_rows -
        clean_rows
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Raw Rows",
        f"{raw_rows:,}"
    )

    c2.metric(
        "Rows Used",
        f"{clean_rows:,}"
    )

    c3.metric(
        "Excluded Rows",
        f"{excluded_rows:,}"
    )

    st.markdown(
        "### Detected Column Mapping"
    )

    mapping_table = pd.DataFrame(
        [
            (
                key,
                value
            )
            for key, value in mapping.items()
        ],
        columns=[
            "Standard Field",
            "Detected Column"
        ]
    )

    st.dataframe(
        mapping_table,
        use_container_width=True,
        hide_index=True
    )

    st.markdown(
        "### RFM Methodology"
    )

    st.markdown(
        """
        **Recency**

        Number of days since the customer's most recent
        valid purchase.

        **Frequency**

        Number of unique orders/invoices made by the customer.

        **Monetary**

        Total customer revenue:

        `Quantity × Unit Price`

        **RFM Score**

        Each metric receives a score from 1 to 5 using
        quintile-based scoring.

        Recency is reversed because fewer days means
        stronger recent engagement.

        **Risk Score**

        The risk score is an explainable analytical proxy
        based primarily on purchase timing, with additional
        consideration for frequency and monetary value.

        **CLV Proxy**

        Average Order Value × annualized order rate ×
        1.5-year planning horizon.

        This is a planning metric and not a guaranteed
        financial forecast.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

