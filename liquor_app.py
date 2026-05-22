import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.linear_model import LinearRegression

# ── Maroon palette ────────────────────────────────────────────────────────────
M1  = "#800000"   # maroon utama
M2  = "#a52a2a"   # maroon medium (brown-red)
M3  = "#c0392b"   # merah tua aksen
M4  = "#d4a5a5"   # maroon muda / fill
M5  = "#f9f0f0"   # background card
M6  = "#4a0000"   # maroon gelap (sidebar / header)
ACC = "#e8c4c4"   # aksen terang
NEG = "#c0392b"   # negatif (merah)
POS = "#6d4c41"   # positif (coklat tua)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Iowa Liquor Sales Dashboard",
    page_icon="🥃",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
    /* Background utama */
    .stApp {{ background-color: #fdf5f5; }}

    /* Sidebar */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {M6} 0%, {M1} 100%);
    }}
    [data-testid="stSidebar"] * {{ color: #f5e6e6 !important; }}
    [data-testid="stSidebar"] .stMultiSelect span {{ background-color: {M2} !important; }}

    /* Divider */
    hr {{ border-color: {M4} !important; }}

    /* Metric card */
    .metric-card {{
        background: white;
        border-radius: 12px;
        padding: 20px 24px;
        box-shadow: 0 2px 10px rgba(128,0,0,0.10);
        border-left: 5px solid {M1};
        margin-bottom: 8px;
    }}
    .metric-label  {{ font-size: 13px; color: #7a4444; font-weight: 500; margin-bottom: 4px; }}
    .metric-value  {{ font-size: 28px; font-weight: 700; color: {M6}; }}
    .metric-delta  {{ font-size: 12px; color: {POS}; margin-top: 4px; }}

    /* Section title */
    .section-title {{
        font-size: 20px; font-weight: 700; color: {M6};
        margin: 24px 0 16px 0; padding-bottom: 8px;
        border-bottom: 3px solid {M1};
    }}

    /* Tab styling */
    .stTabs [data-baseweb="tab"] {{
        color: {M1} !important;
        font-weight: 600;
    }}
    .stTabs [aria-selected="true"] {{
        border-bottom: 3px solid {M1} !important;
        color: {M6} !important;
    }}

    /* Dataframe header */
    [data-testid="stDataFrame"] th {{
        background-color: {M1} !important;
        color: white !important;
    }}

    /* Slider accent */
    [data-testid="stSlider"] > div > div > div {{
        background: {M1} !important;
    }}
</style>
""", unsafe_allow_html=True)

PLOT_BG   = "white"
PAPER_BG  = "white"
GRID_CLR  = "#f5e6e6"
FONT_CLR  = M6

def fig_base(fig, height=400):
    fig.update_layout(
        height=height,
        plot_bgcolor=PLOT_BG,
        paper_bgcolor=PAPER_BG,
        font=dict(color=FONT_CLR, family="sans-serif"),
        margin=dict(t=40, b=10, l=10, r=10),
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID_CLR, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=GRID_CLR, zeroline=False)
    return fig


# ── Data loader ───────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data.csv", parse_dates=["date"])
    df["month_name"] = pd.Categorical(
        df["month_name"],
        categories=["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"],
        ordered=True,
    )
    return df

df_all = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"## 🥃 Iowa Liquor Sales")
    st.markdown("**Dashboard 2026**")
    st.divider()
    st.markdown("### 🔍 Filter Data")

    months_available = sorted(df_all["month"].unique())
    month_labels = {1:"Januari",2:"Februari",3:"Maret",4:"April",
                    5:"Mei",6:"Juni",7:"Juli",8:"Agustus",
                    9:"September",10:"Oktober",11:"November",12:"Desember"}
    selected_months = st.multiselect(
        "Bulan", options=months_available, default=months_available,
        format_func=lambda x: month_labels.get(x, str(x)),
    )
    selected_cities = st.multiselect(
        "Kota", options=sorted(df_all["city"].unique()),
        default=sorted(df_all["city"].unique()),
    )
    selected_cats = st.multiselect(
        "Kategori Produk", options=sorted(df_all["category_name"].unique()),
        default=sorted(df_all["category_name"].unique()),
    )
    st.divider()
    st.caption("Data: BigQuery Iowa Liquor Sales 2026 (757,888 transaksi asli)")

# ── Apply filters ─────────────────────────────────────────────────────────────
df = df_all[
    df_all["month"].isin(selected_months) &
    df_all["city"].isin(selected_cities) &
    df_all["category_name"].isin(selected_cats)
].copy()

if df.empty:
    st.warning("Tidak ada data yang sesuai dengan filter.")
    st.stop()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(f"<h1 style='color:{M6}'>🥃 Iowa Liquor Sales Dashboard</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='color:{M2}'><b>Analisis Penjualan Minuman Keras Iowa — 2026</b> | Data: BigQuery Public Dataset</p>",
            unsafe_allow_html=True)
st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — KPI CARDS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">📊 Key Performance Indicators</div>', unsafe_allow_html=True)

SCALE       = 757888 / len(df_all)
total_sales = df["sale_dollars"].sum() * SCALE
total_bottles = df["bottles_sold"].sum() * SCALE
total_txn   = len(df) * SCALE
avg_sale    = df["sale_dollars"].mean()
avg_margin  = df["margin_pct"].mean()

col1, col2, col3, col4, col5 = st.columns(5)

CARD_COLORS = [M1, M2, M3, "#6d2b2b", "#4a0000"]

def metric_card(col, label, value, border_color=M1):
    with col:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color:{border_color}">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>""", unsafe_allow_html=True)

metric_card(col1, "Total Pendapatan",        f"${total_sales/1e6:.1f}M",  CARD_COLORS[0])
metric_card(col2, "Total Botol Terjual",     f"{total_bottles/1e6:.1f}M", CARD_COLORS[1])
metric_card(col3, "Jumlah Transaksi",        f"{total_txn/1e3:.0f}K",     CARD_COLORS[2])
metric_card(col4, "Rata-rata per Transaksi", f"${avg_sale:.2f}",          CARD_COLORS[3])
metric_card(col5, "Rata-rata Margin",        f"{avg_margin:.1f}%",        CARD_COLORS[4])
st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — TREN PENJUALAN
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">📈 Tren Penjualan Bulanan</div>', unsafe_allow_html=True)

df_monthly = (
    df.groupby(["month","month_name"])
    .agg(total_sales=("sale_dollars","sum"), n_txn=("sale_dollars","count"),
         total_bottles=("bottles_sold","sum"), avg_sales=("sale_dollars","mean"))
    .reset_index().sort_values("month")
)
df_monthly["mom_growth"]          = df_monthly["total_sales"].pct_change() * 100
df_monthly["cumulative"]          = df_monthly["total_sales"].cumsum()
df_monthly["total_sales_scaled"]  = df_monthly["total_sales"] * SCALE
df_monthly["total_bottles_scaled"]= df_monthly["total_bottles"] * SCALE

fig_trend = make_subplots(
    rows=2, cols=2,
    subplot_titles=("Total Penjualan per Bulan (USD)", "Jumlah Transaksi per Bulan",
                    "MoM Growth Rate (%)",             "Penjualan Kumulatif (USD)"),
    vertical_spacing=0.14,
)

fig_trend.add_trace(go.Scatter(
    x=df_monthly["month_name"], y=df_monthly["total_sales_scaled"]/1e6,
    mode="lines+markers+text",
    text=[f"${v:.1f}M" for v in df_monthly["total_sales_scaled"]/1e6],
    textposition="top center",
    line=dict(color=M1, width=2.5), marker=dict(size=8),
    fill="tozeroy", fillcolor="rgba(128,0,0,0.10)", name="Total Sales",
), row=1, col=1)

fig_trend.add_trace(go.Bar(
    x=df_monthly["month_name"], y=df_monthly["n_txn"]*SCALE/1e3,
    marker_color=M4, name="Transaksi (K)",
    text=[f"{v:.0f}K" for v in df_monthly["n_txn"]*SCALE/1e3],
    textposition="outside",
), row=1, col=2)

colors_mom = [NEG if v < 0 else POS for v in df_monthly["mom_growth"].fillna(0)]
fig_trend.add_trace(go.Bar(
    x=df_monthly["month_name"], y=df_monthly["mom_growth"].fillna(0),
    marker_color=colors_mom, name="MoM Growth (%)",
    text=[f"{v:.1f}%" for v in df_monthly["mom_growth"].fillna(0)],
    textposition="outside",
), row=2, col=1)
fig_trend.add_hline(y=0, line_color=M6, line_width=0.8, row=2, col=1)

fig_trend.add_trace(go.Scatter(
    x=df_monthly["month_name"], y=df_monthly["cumulative"]*SCALE/1e6,
    mode="lines+markers",
    line=dict(color=M2, width=2.5),
    fill="tozeroy", fillcolor="rgba(165,42,42,0.10)",
    marker=dict(size=8, symbol="square"), name="Kumulatif",
), row=2, col=2)

fig_trend.update_layout(
    height=520, showlegend=False,
    plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
    font=dict(color=FONT_CLR),
    margin=dict(t=40, b=10),
)
fig_trend.update_xaxes(showgrid=True, gridcolor=GRID_CLR)
fig_trend.update_yaxes(showgrid=True, gridcolor=GRID_CLR)
st.plotly_chart(fig_trend, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — TOP KATEGORI
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">🏷️ Analisis Kategori Produk</div>', unsafe_allow_html=True)

df_cat = (
    df.groupby("category_name")
    .agg(total_sales=("sale_dollars","sum"), n_txn=("sale_dollars","count"),
         avg_sales=("sale_dollars","mean"), total_bottles=("bottles_sold","sum"))
    .reset_index().sort_values("total_sales", ascending=False)
)
df_cat["share_pct"] = df_cat["total_sales"] / df_cat["total_sales"].sum() * 100

col_left, col_right = st.columns([3, 2])

with col_left:
    top_n = st.slider("Tampilkan Top N Kategori", 5, min(20, len(df_cat)), 10)
    df_cat_top = df_cat.head(top_n).sort_values("total_sales")
    fig_cat = go.Figure(go.Bar(
        x=df_cat_top["total_sales"]/1e3, y=df_cat_top["category_name"],
        orientation="h",
        marker=dict(color=df_cat_top["total_sales"],
                    colorscale=[[0,"#f5c6c6"],[0.5,M2],[1,M6]],
                    showscale=False),
        text=[f"${v/1e3:.1f}K ({p:.1f}%)"
              for v,p in zip(df_cat_top["total_sales"], df_cat_top["share_pct"])],
        textposition="outside",
    ))
    fig_base(fig_cat, 420)
    fig_cat.update_layout(title="Total Penjualan per Kategori",
                          xaxis_title="Total Sales (Ribu USD)",
                          margin=dict(l=10,r=130,t=40,b=10))
    st.plotly_chart(fig_cat, use_container_width=True)

with col_right:
    maroon_palette = [M6, M1, M2, M3, "#8b1a1a","#b22222","#cd5c5c","#dc143c"]
    fig_pie = px.pie(
        df_cat.head(8), values="total_sales", names="category_name",
        title="Proporsi Revenue (Top 8 Kategori)",
        color_discrete_sequence=maroon_palette, hole=0.4,
    )
    fig_pie.update_traces(textposition="inside", textinfo="percent+label",
                          textfont_size=10)
    fig_pie.update_layout(height=420, margin=dict(t=40,b=10),
                          showlegend=False,
                          paper_bgcolor=PAPER_BG, font=dict(color=FONT_CLR))
    st.plotly_chart(fig_pie, use_container_width=True)

st.markdown("#### Rata-rata Sales per Transaksi per Kategori (Top 15 tertinggi)")
df_cat_avg = df_cat.sort_values("avg_sales", ascending=False).head(15).sort_values("avg_sales")
fig_avg = go.Figure(go.Bar(
    x=df_cat_avg["avg_sales"], y=df_cat_avg["category_name"],
    orientation="h",
    marker=dict(color=df_cat_avg["avg_sales"],
                colorscale=[[0,ACC],[1,M1]], showscale=False),
    text=[f"${v:.0f}" for v in df_cat_avg["avg_sales"]],
    textposition="outside",
))
fig_base(fig_avg, 380)
fig_avg.update_layout(xaxis_title="Avg Sales per Transaksi (USD)",
                      margin=dict(l=10,r=80,t=10,b=10))
st.plotly_chart(fig_avg, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — GEOGRAFIS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">🗺️ Analisis Geografis (Kota & Toko)</div>', unsafe_allow_html=True)

df_city = (
    df.groupby("city")
    .agg(total_revenue=("sale_dollars","sum"), n_txn=("sale_dollars","count"),
         avg_rev=("sale_dollars","mean"))
    .reset_index().sort_values("total_revenue", ascending=False)
)
df_city["share_pct"] = df_city["total_revenue"] / df_city["total_revenue"].sum() * 100

df_store = (
    df.groupby("store_name")
    .agg(total_revenue=("sale_dollars","sum"), n_txn=("sale_dollars","count"))
    .reset_index().sort_values("total_revenue", ascending=False)
)
df_store["share_pct"] = df_store["total_revenue"] / df_store["total_revenue"].sum() * 100

c1, c2 = st.columns(2)

with c1:
    top15_city = df_city.head(15).sort_values("total_revenue")
    fig_city_chart = go.Figure(go.Bar(
        x=top15_city["total_revenue"]/1e3, y=top15_city["city"],
        orientation="h", marker_color=M4,
        text=[f"${v/1e3:.1f}K ({p:.1f}%)"
              for v,p in zip(top15_city["total_revenue"], top15_city["share_pct"])],
        textposition="outside",
    ))
    fig_base(fig_city_chart, 480)
    fig_city_chart.update_layout(title="Top 15 Kota — Total Revenue",
                                 xaxis_title="Revenue (Ribu USD)",
                                 margin=dict(l=10,r=110,t=40,b=10))
    st.plotly_chart(fig_city_chart, use_container_width=True)

with c2:
    top15_store = df_store.head(15).sort_values("total_revenue")
    fig_store_chart = go.Figure(go.Bar(
        x=top15_store["total_revenue"]/1e3, y=top15_store["store_name"],
        orientation="h", marker_color=M2,
        text=[f"${v/1e3:.1f}K ({p:.1f}%)"
              for v,p in zip(top15_store["total_revenue"], top15_store["share_pct"])],
        textposition="outside",
    ))
    fig_base(fig_store_chart, 480)
    fig_store_chart.update_layout(title="Top 15 Toko — Total Revenue",
                                  xaxis_title="Revenue (Ribu USD)",
                                  margin=dict(l=10,r=110,t=40,b=10))
    st.plotly_chart(fig_store_chart, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — DISTRIBUSI & OUTLIER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">📦 Distribusi & Analisis Outlier</div>', unsafe_allow_html=True)

c1, c2 = st.columns(2)

with c1:
    clip_pct = st.slider("Clip percentile (untuk histogram)", 90, 100, 99, key="clip")
    clip_val = df["sale_dollars"].quantile(clip_pct / 100)
    data_clipped = df["sale_dollars"].clip(upper=clip_val)
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=data_clipped, nbinsx=50,
        marker_color=M4, marker_line_color="white", marker_line_width=0.5,
    ))
    fig_hist.add_vline(x=df["sale_dollars"].median(), line_dash="dash",
                       line_color=M1,
                       annotation_text=f"Median ${df['sale_dollars'].median():.0f}")
    fig_hist.add_vline(x=df["sale_dollars"].mean(), line_dash="dash",
                       line_color=M3,
                       annotation_text=f"Mean ${df['sale_dollars'].mean():.0f}")
    fig_base(fig_hist, 350)
    fig_hist.update_layout(title=f"Distribusi Sale Dollars (dipotong di P{clip_pct})",
                           xaxis_title="Sale Dollars (USD)", yaxis_title="Frekuensi")
    st.plotly_chart(fig_hist, use_container_width=True)

with c2:
    num_col = st.selectbox("Pilih kolom untuk boxplot",
                           ["sale_dollars","bottles_sold","state_bottle_retail",
                            "volume_sold_liters","margin_pct","price_per_liter"])
    fig_box = go.Figure(go.Box(
        y=df[num_col], name=num_col,
        marker_color=M1, boxmean="sd",
        fillcolor="rgba(212,165,165,0.4)", line_color=M1,
    ))
    fig_base(fig_box, 350)
    fig_box.update_layout(title=f"Boxplot: {num_col}", yaxis_title=num_col)
    st.plotly_chart(fig_box, use_container_width=True)

st.markdown("#### Deteksi Outlier — Metode IQR")
cols_iqr = ["sale_dollars","bottles_sold","state_bottle_retail",
            "volume_sold_liters","state_bottle_cost","margin_pct"]
iqr_rows = []
for col in cols_iqr:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lo, hi = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    n_out = ((df[col] < lo) | (df[col] > hi)).sum()
    iqr_rows.append({"Kolom":col,"Q1":round(Q1,2),"Q3":round(Q3,2),"IQR":round(IQR,2),
                     "Batas Bawah":round(lo,2),"Batas Atas":round(hi,2),
                     "Jml Outlier":n_out,"Persen (%)":round(n_out/len(df)*100,2)})
st.dataframe(pd.DataFrame(iqr_rows), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — KORELASI & REGRESI
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">🔗 Korelasi & Analisis Regresi</div>', unsafe_allow_html=True)

c1, c2 = st.columns([2, 3])
corr_cols = ["sale_dollars","bottles_sold","state_bottle_retail",
             "volume_sold_liters","margin_pct","price_per_liter"]

with c1:
    st.markdown("#### Heatmap Korelasi Pearson")
    corr_matrix = df[corr_cols].corr().round(4)
    fig_heat = go.Figure(go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns.tolist(),
        y=corr_matrix.index.tolist(),
        colorscale=[[0,"#f5e6e6"],[0.5,M4],[1,M6]],
        zmin=-1, zmax=1,
        text=corr_matrix.values.round(2),
        texttemplate="%{text}",
        textfont={"size":11,"color":"white"},
    ))
    fig_heat.update_layout(height=380, margin=dict(t=10,b=10,l=10,r=10),
                           paper_bgcolor=PAPER_BG,
                           font=dict(color=FONT_CLR),
                           xaxis=dict(tickangle=-30))
    st.plotly_chart(fig_heat, use_container_width=True)

with c2:
    st.markdown("#### Scatterplot & Regresi Linear")
    sc_x = st.selectbox("Variabel X", corr_cols, index=1)
    sc_y = st.selectbox("Variabel Y", corr_cols, index=0)

    df_sc = df.sample(n=min(1000, len(df)), random_state=42)
    r, p_val = stats.pearsonr(df[sc_x], df[sc_y])
    X = df[[sc_x]].values; y_arr = df[sc_y].values
    reg = LinearRegression().fit(X, y_arr)
    r2 = reg.score(X, y_arr)
    coef = reg.coef_[0]; intercept = reg.intercept_
    x_range = np.linspace(X.min(), X.max(), 200)
    y_range = reg.predict(x_range.reshape(-1,1))

    interp = "kuat" if abs(r)>=0.7 else "sedang" if abs(r)>=0.4 else "lemah"
    fig_sc = go.Figure()
    fig_sc.add_trace(go.Scatter(
        x=df_sc[sc_x], y=df_sc[sc_y], mode="markers",
        marker=dict(color=M1, opacity=0.35, size=5), name="Data",
    ))
    fig_sc.add_trace(go.Scatter(
        x=x_range, y=y_range, mode="lines",
        line=dict(color=M3, width=2.5),
        name=f"Regresi: y={coef:.2f}x+{intercept:.2f}",
    ))
    fig_sc.add_annotation(
        x=0.03, y=0.97, xref="paper", yref="paper",
        text=f"r = {r:.4f} ({interp})<br>R² = {r2:.4f}<br>p-value = {p_val:.2e}",
        showarrow=False, align="left",
        bgcolor="white", bordercolor=M1, borderwidth=1,
        font=dict(size=11, color=M6),
    )
    fig_base(fig_sc, 380)
    fig_sc.update_layout(xaxis_title=sc_x, yaxis_title=sc_y, showlegend=True)
    st.plotly_chart(fig_sc, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — STATISTIK & DATA MENTAH
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">📋 Statistik Deskriptif & Data Mentah</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📊 Statistik Numerik", "🏷️ Statistik Kategorik", "🗃️ Data Mentah"])

with tab1:
    st.dataframe(df.select_dtypes(include=np.number).describe().round(3).T,
                 use_container_width=True)

with tab2:
    cat_summary = []
    for col in ["city","county","category_name","vendor_name","store_name"]:
        top  = df[col].value_counts().index[0]
        freq = df[col].value_counts().iloc[0]
        cat_summary.append({"Kolom":col,"Nilai Unik":df[col].nunique(),
                            "Terbanyak":top,"Frekuensi":freq,
                            "Persen (%)":round(freq/len(df)*100,2)})
    st.dataframe(pd.DataFrame(cat_summary), use_container_width=True, hide_index=True)

with tab3:
    max_rows = st.slider("Tampilkan N baris", 10, 200, 50)
    show_cols = st.multiselect(
        "Pilih kolom",
        df.columns.tolist(),
        default=["date","city","category_name","store_name",
                 "vendor_name","bottles_sold","sale_dollars","margin_pct"],
    )
    st.dataframe(df[show_cols].head(max_rows), use_container_width=True, hide_index=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    f"<center><small style='color:{M2}'>Iowa Liquor Sales Dashboard · "
    "Data: BigQuery Public Dataset 2026 · Dibuat dengan Streamlit & Plotly</small></center>",
    unsafe_allow_html=True,
)
