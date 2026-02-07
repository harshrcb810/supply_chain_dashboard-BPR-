import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="Integrated Supply Chain & BPR Dashboard",
    page_icon="📊",
    layout="wide"
)

# ===============================
# Custom CSS
# ===============================
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .data-quality-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<div class="main-header">🏭 Integrated Supply Chain & BPR Analytics Dashboard</div>', unsafe_allow_html=True)


# ===============================
# Robust Date Parsing Helper
# ===============================
def parse_date_column(series):
    if series.dropna().empty:
        return series, 0, 0

    cleaned = series.astype(str).str.strip()

    date_formats = [
        '%d-%m-%Y', '%d/%m/%Y', '%Y-%m-%d', '%m-%d-%Y',
        '%d-%b-%Y', '%d %b %Y', '%Y%m%d', '%d-%m-%y',
    ]

    best_result = None
    best_count = 0

    for fmt in date_formats:
        try:
            parsed = pd.to_datetime(cleaned, format=fmt, errors='coerce')
            valid_count = parsed.notna().sum()
            if valid_count > best_count:
                best_count = valid_count
                best_result = parsed
        except Exception:
            continue

    if best_result is None or best_count == 0:
        best_result = pd.to_datetime(cleaned, dayfirst=True, errors='coerce')
        best_count = best_result.notna().sum()

    total = len(series.dropna())
    failed = total - best_count
    return best_result, best_count, failed


# ===============================
# Dynamic Color Detection — reads from data, no hardcoding
# ===============================
def get_all_colors(df, color_column='On hand Inv. Color'):
    if color_column not in df.columns:
        return []

    raw_colors = df[color_column].dropna().unique().tolist()
    normalized = sorted(list(set(
        c.strip().title() for c in raw_colors
        if isinstance(c, str) and c.strip() and c.strip().title() != 'Nan'
    )))
    return normalized


# ===============================
# Penetration Calculation — ((Norm - Stock) / Norm) * 100
# ===============================
def calc_penetration(norm_val, stock_val):
    """Calculate buffer penetration: ((Norm - Stock) / Norm) * 100"""
    if norm_val == 0:
        return 0.0
    return float(((norm_val - stock_val) / norm_val) * 100)


def calc_penetration_series(df):
    """Add a calculated Penetration % column to a dataframe that has Norm and Stock."""
    norm = df['Norm'].astype(float)
    stock = df['Stock'].astype(float)
    pen = ((norm - stock) / norm.replace(0, np.nan)) * 100
    return pen.fillna(0)


# ===============================
# Sanitize DataFrame for Streamlit display
# ===============================
def sanitize_for_display(df):
    df = df.copy()
    df.attrs = {}
    df = df.reset_index(drop=True)

    for col in df.columns:
        df[col] = df[col].apply(
            lambda x: str(x) if pd.notna(x) and x is not None else ''
        )
    return df


# ===============================
# Deduplication Helper
# ===============================
def deduplicate_bpr(df):
    key_cols = [col for col in ['SKUCode', 'Location Code', 'Date'] if col in df.columns]
    if not key_cols:
        return df, 0

    before = len(df)
    df_dedup = df.drop_duplicates(subset=key_cols, keep='last')
    removed = before - len(df_dedup)
    return df_dedup, removed


# ===============================
# Load Data Functions
# ===============================
@st.cache_data
def load_bpr_data(file_path):
    df = pd.read_csv(file_path, encoding='latin-1', low_memory=False)
    df.columns = df.columns.str.strip()

    if 'Location Type' in df.columns:
        df['Location Type'] = df['Location Type'].astype(str).str.strip().str.title()

    for color_col in ['On hand Inv. Color', 'Pipeline Inv. Color']:
        if color_col in df.columns:
            df[color_col] = df[color_col].astype(str).str.strip().str.title()
            df[color_col] = df[color_col].replace('Nan', pd.NA)

    # Convert ALL numeric-looking columns robustly
    for col in df.columns:
        if col in ['SKUCode', 'Description', 'Location Code', 'Location Description',
                    'On hand Inv. Color', 'Pipeline Inv. Color', 'Latest Remark',
                    'SourceCode', 'SourceLocation', 'Location Type', 'Tags',
                    'Tyre Type', 'Construction', 'Top SKU', 'Trigger', 'Date']:
            continue
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace('%', '', regex=False).str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Parse Date
    date_valid = 0
    date_failed = 0
    if 'Date' in df.columns:
        df['Date'], date_valid, date_failed = parse_date_column(df['Date'])

    # Deduplicate
    df, dup_count = deduplicate_bpr(df)

    # Recalculate penetration from formula: ((Norm - Stock) / Norm) * 100
    if 'Norm' in df.columns and 'Stock' in df.columns:
        df['Calc Penetration %'] = calc_penetration_series(df)

    df.attrs['_date_valid'] = int(date_valid)
    df.attrs['_date_failed'] = int(date_failed)
    df.attrs['_duplicates_removed'] = int(dup_count)

    return df


@st.cache_data
def load_supply_chain_data():
    try:
        sales = pd.read_csv("daily_sales.csv")
        sales["invoice_date"] = pd.to_datetime(sales["invoice_date"])

        prod = pd.read_csv("dly_prod_jan.csv")
        prod["Prod.Date"] = pd.to_datetime(prod["Prod.Date"])

        stock = pd.read_csv("daily_stock.csv")
        stock["date"] = pd.to_datetime(stock["date"].astype(str), format='%Y%m%d')

        stock_btp = pd.read_csv("daily_stock_btp.csv")
        stock_btp["date"] = pd.to_datetime(stock_btp["date"].astype(str), format='%Y%m%d')

        demand_summary = pd.read_excel("demand_summary.xlsx")

        datewise_stock = pd.read_excel("datewise_comparison.xlsx", sheet_name="Stock")
        datewise_norm = pd.read_excel("datewise_comparison.xlsx", sheet_name="Norm")
        datewise_vnorm = pd.read_excel("datewise_comparison.xlsx", sheet_name="Virtual Norm")
        datewise_requirement = pd.read_excel("datewise_comparison.xlsx", sheet_name="Requirement")

        for sheet in [datewise_stock, datewise_norm, datewise_vnorm, datewise_requirement]:
            sheet['Market'] = sheet['SKU+Market'].str.split('_').str[-1]

        return {
            'sales': sales, 'prod': prod, 'stock': stock, 'stock_btp': stock_btp,
            'demand_summary': demand_summary, 'datewise_stock': datewise_stock,
            'datewise_norm': datewise_norm, 'datewise_vnorm': datewise_vnorm,
            'datewise_requirement': datewise_requirement
        }
    except Exception:
        return None


# ===============================
# Helper: Get penetration color bucket from calculated %
# ===============================
def get_color_for_penetration(pen_pct):
    """
    Determine status color from calculated penetration %.
    Formula: ((Norm - Stock) / Norm) * 100

    Color ranges (from user requirement):
        100%          → Black  (Stockout: Stock = 0)
        67% - 100%    → Red    (Critical)
        33% - 67%     → Yellow (Warning)
        0%  - 33%     → Green  (Good)
        < 0%          → White  (Overstock: Stock > Norm)
    """
    if pen_pct >= 100:
        return 'Black'
    elif pen_pct >= 67:
        return 'Red'
    elif pen_pct >= 33:
        return 'Yellow'
    elif pen_pct >= 0:
        return 'Green'
    else:
        return 'White'


def get_display_props_for_penetration(pen_pct):
    """
    Get display properties (color, text, icon) based on penetration %.
    Directly maps the calculated penetration value to the correct visual style.
    """
    color_name = get_color_for_penetration(pen_pct)

    props = {
        'Black':  {'color': '#1a1a2e', 'text': 'STOCKOUT',   'icon': '⚫', 'rgb': 'rgba(26, 26, 46, 0.9)'},
        'Red':    {'color': '#c0392b', 'text': 'CRITICAL',   'icon': '🔴', 'rgb': 'rgba(192, 57, 43, 0.85)'},
        'Yellow': {'color': '#f39c12', 'text': 'WARNING',    'icon': '🟡', 'rgb': 'rgba(243, 156, 18, 0.85)'},
        'Green':  {'color': '#27ae60', 'text': 'GOOD',       'icon': '🟢', 'rgb': 'rgba(39, 174, 96, 0.85)'},
        'White':  {'color': '#7f8c8d', 'text': 'OVERSTOCK',  'icon': '⚪', 'rgb': 'rgba(149, 165, 166, 0.85)'},
    }

    return props.get(color_name, props['Green'])


def get_display_props(color_name):
    """Map a data color name to display properties for charts (used in non-penetration contexts)."""
    mapping = {
        'Black':  {'color': '#1a1a2e', 'text': 'STOCKOUT',       'icon': '⚫', 'rgb': 'rgba(26, 26, 46, 0.9)'},
        'Green':  {'color': '#27ae60', 'text': 'GOOD',           'icon': '🟢', 'rgb': 'rgba(39, 174, 96, 0.85)'},
        'Yellow': {'color': '#f39c12', 'text': 'WARNING',        'icon': '🟡', 'rgb': 'rgba(243, 156, 18, 0.85)'},
        'Orange': {'color': '#e67e22', 'text': 'LOW INVENTORY',  'icon': '🟠', 'rgb': 'rgba(230, 126, 34, 0.85)'},
        'Red':    {'color': '#c0392b', 'text': 'CRITICAL',       'icon': '🔴', 'rgb': 'rgba(192, 57, 43, 0.85)'},
        'Blue':   {'color': '#3498db', 'text': 'OVER STOCKED',   'icon': '🔵', 'rgb': 'rgba(52, 152, 219, 0.85)'},
        'White':  {'color': '#7f8c8d', 'text': 'OVERSTOCK',      'icon': '⚪', 'rgb': 'rgba(149, 165, 166, 0.85)'},
    }
    return mapping.get(
        color_name,
        {'color': '#7f8c8d', 'text': str(color_name).upper(), 'icon': '⬜', 'rgb': 'rgba(127, 140, 141, 0.8)'}
    )


# ===============================
# Try Loading BPR Data
# ===============================
bpr_df = None
possible_files = [r'BufferPenetration_Combined.csv']

for file_path in possible_files:
    try:
        bpr_df = load_bpr_data(file_path)
        st.success(f"✅ BPR Data loaded from: {file_path}")

        date_valid = bpr_df.attrs.get('_date_valid', 0)
        date_failed = bpr_df.attrs.get('_date_failed', 0)
        dup_removed = bpr_df.attrs.get('_duplicates_removed', 0)

        quality_msgs = []
        if date_failed > 0:
            quality_msgs.append(f"⚠️ {date_failed} date(s) could not be parsed and were excluded")
        if dup_removed > 0:
            quality_msgs.append(f"🔄 {dup_removed} duplicate row(s) removed (same SKU + Location + Date)")

        if quality_msgs:
            with st.expander("📋 Data Quality Report", expanded=False):
                for msg in quality_msgs:
                    st.warning(msg)
                st.info(f"✅ {date_valid} dates parsed successfully | 📊 {len(bpr_df):,} total records after cleanup")
        break
    except Exception:
        continue

supply_chain_data = load_supply_chain_data()

# ===============================
# Sidebar - Dashboard Selection
# ===============================
st.sidebar.header("📊 Dashboard Selection")

dashboard_options = []
if bpr_df is not None:
    dashboard_options.extend(["BPR Analysis", "BPR Summary by Location & Color"])
if supply_chain_data is not None:
    dashboard_options.append("Supply Chain Analytics")

if not dashboard_options:
    st.error("❌ No data files found. Please upload BPR CSV or ensure supply chain data files are available.")
    uploaded = st.sidebar.file_uploader("Upload BPR CSV file", type=['csv'])
    if uploaded:
        bpr_df = load_bpr_data(uploaded)
        st.success("✅ File uploaded successfully!")
        dashboard_options.extend(["BPR Analysis", "BPR Summary by Location & Color"])
    else:
        st.stop()

selected_dashboard = st.sidebar.selectbox("Select Dashboard", dashboard_options)
st.sidebar.markdown("---")


# ===============================
# DASHBOARD 1: BPR ANALYSIS
# ===============================
if selected_dashboard == "BPR Analysis" and bpr_df is not None:
    st.header("📦 Buffer Penetration Report (BPR) Analysis")

    st.sidebar.header("🔍 Select Filters")
    st.sidebar.subheader("📅 Date Range")

    start_date = None
    end_date = None
    date_suffix = "all"

    if 'Date' in bpr_df.columns and bpr_df['Date'].notna().any():
        min_date = bpr_df['Date'].min()
        max_date = bpr_df['Date'].max()

        st.sidebar.caption(f"Data range: {min_date.strftime('%d %b %Y')} → {max_date.strftime('%d %b %Y')}")

        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key='bpr_date_range'
        )

        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
        else:
            start_date = end_date = date_range

        df = bpr_df[
            (bpr_df['Date'].dt.normalize() >= pd.Timestamp(start_date).normalize()) &
            (bpr_df['Date'].dt.normalize() <= pd.Timestamp(end_date).normalize())
        ].copy()

        date_suffix = start_date.strftime("%Y%m%d")
        st.sidebar.info(f"📊 Records: {len(df):,}")
        st.sidebar.caption(f"From: {start_date.strftime('%d %b %Y')}")
        st.sidebar.caption(f"To: {end_date.strftime('%d %b %Y')}")

        total_records = len(bpr_df)
        filtered_records = len(df)
        if filtered_records < total_records * 0.1 and total_records > 100:
            st.sidebar.warning(
                f"⚠️ Only {filtered_records}/{total_records} records match this date range."
            )
    else:
        df = bpr_df.copy()
        st.sidebar.warning("⚠️ No date information available — showing all records")

    st.sidebar.markdown("---")

    # SKU selection — values from data
    st.sidebar.subheader("🔍 SKU Selection")
    skus = sorted(df['SKUCode'].dropna().unique().tolist())

    if len(skus) == 0:
        st.warning("⚠️ No SKU data available for the selected filters")
        st.info("Please adjust your filters in the sidebar.")
        st.stop()

    selected_sku = st.sidebar.selectbox("Select SKU", skus, key='bpr_sku')
    st.markdown("---")

    # ===============================
    # SECTION 1: ALL LOCATION DETAILS FOR SELECTED SKU
    # ===============================
    st.subheader(f"📦 All Location Details for SKU: {selected_sku}")

    sku_all_locations = df[df['SKUCode'] == selected_sku].copy()

    if len(sku_all_locations) > 0:
        # Colors and location types — from data only
        color_order = get_all_colors(sku_all_locations)
        sku_location_types = sorted(sku_all_locations['Location Type'].dropna().unique())

        st.markdown("#### 📊 Summary: Norm & Stock by Color Status and Location Type")

        location_summary_data = []

        for i, color in enumerate(color_order):
            row_data = {'Row Labels': f"{i + 1}. {color}"}

            for loc_type in sku_location_types:
                filtered = sku_all_locations[
                    (sku_all_locations['On hand Inv. Color'] == color) &
                    (sku_all_locations['Location Type'] == loc_type)
                ]
                row_data[f'Sum of {loc_type} Norm'] = int(filtered['Norm'].sum())
                row_data[f'Sum of {loc_type} Stock'] = int(filtered['Stock'].sum())

            location_summary_data.append(row_data)

        # Unclassified rows (NaN/blank color)
        uncategorized = sku_all_locations[
            sku_all_locations['On hand Inv. Color'].isna() |
            (sku_all_locations['On hand Inv. Color'].astype(str).str.strip() == '') |
            (sku_all_locations['On hand Inv. Color'].astype(str).str.strip() == 'Nan')
        ]
        if len(uncategorized) > 0:
            row_data = {'Row Labels': f'{len(color_order) + 1}. (Unclassified)'}
            for loc_type in sku_location_types:
                filtered = uncategorized[uncategorized['Location Type'] == loc_type]
                row_data[f'Sum of {loc_type} Norm'] = int(filtered['Norm'].sum())
                row_data[f'Sum of {loc_type} Stock'] = int(filtered['Stock'].sum())
            location_summary_data.append(row_data)
            st.warning(f"⚠️ {len(uncategorized)} record(s) have no color classification")

        # Grand Total
        grand_total_row = {'Row Labels': 'Grand Total'}
        for loc_type in sku_location_types:
            filtered = sku_all_locations[sku_all_locations['Location Type'] == loc_type]
            grand_total_row[f'Sum of {loc_type} Norm'] = int(filtered['Norm'].sum())
            grand_total_row[f'Sum of {loc_type} Stock'] = int(filtered['Stock'].sum())
        location_summary_data.append(grand_total_row)

        location_summary_df = pd.DataFrame(location_summary_data)
        st.dataframe(sanitize_for_display(location_summary_df), width='stretch', hide_index=True)

        csv_loc_summary = location_summary_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Location Summary as CSV",
            data=csv_loc_summary,
            file_name=f'sku_{selected_sku}_location_summary_{date_suffix}.csv',
            mime='text/csv',
            key='download_loc_summary'
        )

        st.markdown("---")

        # ===============================
        # Detailed Records for Each Location
        # ===============================
        st.markdown("#### 📋 Detailed Records by Location Type")

        if len(sku_location_types) > 0:
            location_tabs = st.tabs([f"📍 {loc}" for loc in sku_location_types])

            for idx, loc_type in enumerate(sku_location_types):
                with location_tabs[idx]:
                    loc_data = sku_all_locations[sku_all_locations['Location Type'] == loc_type].copy()

                    total_norm = float(loc_data['Norm'].sum())
                    total_stock = float(loc_data['Stock'].sum())
                    avg_pen = calc_penetration(total_norm, total_stock)

                    col_m1, col_m2, col_m3, col_m4 = st.columns(4)

                    with col_m1:
                        st.metric("🏢 Total Locations", int(len(loc_data)))
                    with col_m2:
                        st.metric("📦 Total Stock", f"{int(total_stock):,}")
                    with col_m3:
                        st.metric("🎯 Total Norm", f"{int(total_norm):,}")
                    with col_m4:
                        st.metric("📊 Avg Penetration", f"{avg_pen:.1f}%")

                    st.markdown("---")

                    st.markdown(f"**Color Status Distribution in {loc_type}:**")

                    color_dist = loc_data.groupby('On hand Inv. Color').agg({
                        'Stock': 'sum',
                        'Norm': 'sum',
                        'SKUCode': 'count'
                    }).reset_index()
                    color_dist.columns = ['Color Status', 'Total Stock', 'Total Norm', 'Location Count']
                    # Calculate penetration per color group
                    color_dist['Penetration %'] = color_dist.apply(
                        lambda r: f"{calc_penetration(r['Total Norm'], r['Total Stock']):.1f}%", axis=1
                    )
                    for c in ['Total Stock', 'Total Norm', 'Location Count']:
                        color_dist[c] = color_dist[c].astype(int)
                    color_dist = color_dist.sort_values('Total Stock', ascending=False)

                    st.dataframe(sanitize_for_display(color_dist), width='stretch', hide_index=True)

                    st.markdown("---")

                    st.markdown(f"**All {loc_type} Locations for SKU {selected_sku}:**")

                    display_columns = [
                        'Location Code', 'Location Description', 'SourceLocation',
                        'Stock', 'Norm', 'Calc Penetration %', 'On hand Inv. Color',
                        'GIT/ Pending', 'Virtual Norm', 'Pipeline Inv. Pen', 'Pipeline Inv. Color',
                        'Tyre Type', 'Construction', 'Top SKU', 'Date'
                    ]

                    available_columns = [col for col in display_columns if col in loc_data.columns]
                    display_data = loc_data[available_columns].copy()

                    if 'Stock' in display_data.columns:
                        display_data['Stock'] = display_data['Stock'].apply(lambda x: f"{int(x):,}" if pd.notna(x) else '0')
                    if 'Norm' in display_data.columns:
                        display_data['Norm'] = display_data['Norm'].apply(lambda x: f"{int(x):,}" if pd.notna(x) else '0')
                    if 'Calc Penetration %' in display_data.columns:
                        display_data['Calc Penetration %'] = display_data['Calc Penetration %'].apply(
                            lambda x: f"{x:.2f}%" if pd.notna(x) else 'N/A')
                    if 'GIT/ Pending' in display_data.columns:
                        display_data['GIT/ Pending'] = display_data['GIT/ Pending'].apply(
                            lambda x: f"{int(x):,}" if pd.notna(x) else '0')
                    if 'Virtual Norm' in display_data.columns:
                        display_data['Virtual Norm'] = display_data['Virtual Norm'].apply(
                            lambda x: f"{int(x):,}" if pd.notna(x) and x != 0 else '-')
                    if 'Pipeline Inv. Pen' in display_data.columns:
                        display_data['Pipeline Inv. Pen'] = display_data['Pipeline Inv. Pen'].apply(
                            lambda x: f"{x:.2f}%" if pd.notna(x) and x != 0 else 'N/A')
                    if 'Date' in display_data.columns:
                        display_data['Date'] = display_data['Date'].apply(
                            lambda x: x.strftime('%d %b %Y') if pd.notna(x) else 'N/A')

                    st.dataframe(sanitize_for_display(display_data), width='stretch', hide_index=True)

                    csv_loc = loc_data.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label=f"📥 Download {loc_type} Data as CSV",
                        data=csv_loc,
                        file_name=f'sku_{selected_sku}_{loc_type}_{date_suffix}.csv',
                        mime='text/csv',
                        key=f'download_{loc_type}'
                    )

        st.markdown("---")

        # ===============================
        # Location-wise comparison table
        # ===============================
        st.markdown("#### 📋 Location-wise Comparison Table")

        location_comparison = sku_all_locations.groupby(['Location Type', 'On hand Inv. Color']).agg({
            'Stock': 'sum',
            'Norm': 'sum',
            'GIT/ Pending': 'sum',
            'Virtual Norm': 'sum',
            'Location Code': 'count'
        }).reset_index()

        location_comparison.columns = [
            'Location Type', 'Color Status', 'Total Stock', 'Total Norm',
            'Total GIT/Pending', 'Total Virtual Norm', 'Number of Locations'
        ]

        # Calculate penetration per group
        location_comparison['Penetration %'] = location_comparison.apply(
            lambda r: f"{calc_penetration(r['Total Norm'], r['Total Stock']):.1f}%", axis=1
        )

        location_comparison['Total Stock'] = location_comparison['Total Stock'].apply(lambda x: f"{int(x):,}")
        location_comparison['Total Norm'] = location_comparison['Total Norm'].apply(lambda x: f"{int(x):,}")
        location_comparison['Total GIT/Pending'] = location_comparison['Total GIT/Pending'].apply(lambda x: f"{int(x):,}")
        location_comparison['Total Virtual Norm'] = location_comparison['Total Virtual Norm'].apply(
            lambda x: f"{int(x):,}" if x != 0 else '-')

        st.dataframe(sanitize_for_display(location_comparison), width='stretch', hide_index=True)

    else:
        st.warning(f"⚠️ No data found for SKU: {selected_sku}")

    st.markdown("---")

    # ===============================
    # SECTION 2: STOCK VS NORM ANALYSIS FOR ALL LOCATIONS
    # ===============================
    st.subheader(f"📊 Stock vs Norm Analysis - All Locations for SKU: {selected_sku}")

    sku_all_locations_data = df[df['SKUCode'] == selected_sku].copy()

    if len(sku_all_locations_data) > 0:
        all_location_types = sorted(sku_all_locations_data['Location Type'].dropna().unique())

        if len(all_location_types) > 0:
            location_analysis_tabs = st.tabs([f"📍 {loc}" for loc in all_location_types])

            for idx, loc_type in enumerate(all_location_types):
                with location_analysis_tabs[idx]:
                    loc_sku_data = sku_all_locations_data[
                        sku_all_locations_data['Location Type'] == loc_type
                    ].copy()

                    if len(loc_sku_data) > 0:
                        total_stock = float(loc_sku_data['Stock'].sum())
                        total_norm = float(loc_sku_data['Norm'].sum())
                        avg_penetration = calc_penetration(total_norm, total_stock)

                        # Dominant color from data
                        color_counts = loc_sku_data['On hand Inv. Color'].dropna().value_counts()
                        dominant_color = str(color_counts.index[0]) if len(color_counts) > 0 else 'N/A'

                        st.markdown(f"### 📍 {loc_type}")
                        st.markdown(f"**SKU:** {selected_sku}")
                        if 'Description' in loc_sku_data.columns:
                            desc = loc_sku_data['Description'].dropna()
                            if len(desc) > 0:
                                st.markdown(f"**Description:** {desc.iloc[0]}")

                        st.markdown("---")

                        st.markdown("#### 📊 Key Metrics")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("🏢 Locations", int(len(loc_sku_data)))
                        with col2:
                            st.metric("📦 Total Stock", f"{int(total_stock):,}")
                        with col3:
                            st.metric("🎯 Total Norm", f"{int(total_norm):,}")
                        with col4:
                            st.metric("📊 Penetration", f"{avg_penetration:.1f}%")

                        st.markdown("---")
                        st.markdown("#### 📊 Stock vs Norm Comparison")

                        col_chart1, col_chart2 = st.columns(2)

                        with col_chart1:
                            fig_stock_norm = go.Figure()
                            fig_stock_norm.add_trace(go.Bar(
                                name='Stock', x=[f'{loc_type}'], y=[total_stock],
                                marker_color='#3498db',
                                text=[f"{int(total_stock):,}"], textposition='auto', width=0.4
                            ))
                            fig_stock_norm.add_trace(go.Bar(
                                name='Norm', x=[f'{loc_type}'], y=[total_norm],
                                marker_color='#e74c3c',
                                text=[f"{int(total_norm):,}"], textposition='auto', width=0.4
                            ))
                            fig_stock_norm.update_layout(
                                title=f"Stock vs Norm - {loc_type}", height=400,
                                showlegend=True, yaxis_title="Quantity", barmode='group'
                            )
                            st.plotly_chart(fig_stock_norm, use_container_width=True)

                        with col_chart2:
                            penetration_display = avg_penetration
                            # Color based on calculated penetration %, NOT from data column
                            display_props = get_display_props_for_penetration(penetration_display)

                            fig_penetration = go.Figure()
                            max_x = max(120, abs(penetration_display) * 1.3)

                            # --- Colored zone bands showing the ranges ---
                            # White zone: < 0% (overstock)
                            # We don't draw negative, but if pen < 0 the bar color handles it

                            # Green zone: 0% - 33%
                            fig_penetration.add_shape(
                                type="rect", x0=0, x1=33, y0=-0.4, y1=0.4,
                                fillcolor="rgba(39, 174, 96, 0.15)",
                                line=dict(width=0), layer="below"
                            )
                            fig_penetration.add_annotation(
                                x=16.5, y=-0.4, text="Good", showarrow=False,
                                yshift=-14, font=dict(size=9, color='#27ae60')
                            )

                            # Yellow zone: 33% - 67%
                            fig_penetration.add_shape(
                                type="rect", x0=33, x1=67, y0=-0.4, y1=0.4,
                                fillcolor="rgba(243, 156, 18, 0.15)",
                                line=dict(width=0), layer="below"
                            )
                            fig_penetration.add_annotation(
                                x=50, y=-0.4, text="Warning", showarrow=False,
                                yshift=-14, font=dict(size=9, color='#f39c12')
                            )

                            # Red zone: 67% - 100%
                            fig_penetration.add_shape(
                                type="rect", x0=67, x1=100, y0=-0.4, y1=0.4,
                                fillcolor="rgba(192, 57, 43, 0.15)",
                                line=dict(width=0), layer="below"
                            )
                            fig_penetration.add_annotation(
                                x=83.5, y=-0.4, text="Critical", showarrow=False,
                                yshift=-14, font=dict(size=9, color='#c0392b')
                            )

                            # Black zone: 100% (stockout)
                            if max_x > 100:
                                fig_penetration.add_shape(
                                    type="rect", x0=100, x1=max_x, y0=-0.4, y1=0.4,
                                    fillcolor="rgba(26, 26, 46, 0.08)",
                                    line=dict(width=0), layer="below"
                                )
                                fig_penetration.add_annotation(
                                    x=min(110, max_x - 5), y=-0.4, text="Stockout", showarrow=False,
                                    yshift=-14, font=dict(size=9, color='#1a1a2e')
                                )

                            # Zone boundary lines
                            for boundary in [33, 67, 100]:
                                if boundary <= max_x:
                                    fig_penetration.add_shape(
                                        type="line", x0=boundary, x1=boundary, y0=-0.4, y1=0.4,
                                        line=dict(color="rgba(0,0,0,0.2)", width=1, dash="dot"),
                                        layer="below"
                                    )

                            # Background bar
                            fig_penetration.add_trace(go.Bar(
                                y=['Penetration'], x=[max_x], orientation='h',
                                marker=dict(color='rgba(240, 240, 240, 0.2)',
                                            line=dict(color='rgba(200, 200, 200, 0.3)', width=1)),
                                showlegend=False, hoverinfo='skip'
                            ))

                            # Actual penetration bar — colored by penetration value
                            bar_value = abs(penetration_display)
                            # For overstock (negative pen), show small bar with white color
                            if penetration_display < 0:
                                bar_value = min(abs(penetration_display), max_x * 0.3)

                            fig_penetration.add_trace(go.Bar(
                                y=['Penetration'], x=[bar_value], orientation='h',
                                marker=dict(color=display_props['rgb'],
                                            line=dict(color=display_props['color'], width=2)),
                                text=[f"{penetration_display:.1f}%"], textposition='inside',
                                textfont=dict(size=22, color='white', family='Arial Black'),
                                showlegend=False,
                                hovertemplate=(
                                    f"<b>Penetration: {penetration_display:.1f}%</b><br>"
                                    f"Status: {display_props['text']}<br>"
                                    f"Formula: ((Norm−Stock)/Norm)×100<br>"
                                    f"Norm: {int(total_norm):,} | Stock: {int(total_stock):,}"
                                    f"<extra></extra>"
                                )
                            ))

                            # Target line at 100%
                            if max_x >= 100:
                                fig_penetration.add_shape(
                                    type="line", x0=100, x1=100, y0=-0.5, y1=0.5,
                                    line=dict(color="rgba(0, 0, 0, 0.6)", width=3, dash="dash")
                                )
                                fig_penetration.add_annotation(
                                    x=100, y=0.5, text="100%", showarrow=False,
                                    yshift=20, font=dict(size=10, color='#333')
                                )

                            fig_penetration.update_layout(
                                height=400, margin=dict(l=20, r=20, t=80, b=50),
                                xaxis=dict(range=[0, max_x], showgrid=False,
                                           zeroline=False, title='', ticksuffix='%'),
                                yaxis=dict(showticklabels=False, showgrid=False),
                                plot_bgcolor='white', paper_bgcolor='white',
                                title=dict(
                                    text=f"<b>{display_props['icon']} {display_props['text']} — {penetration_display:.1f}%</b>",
                                    font=dict(size=18, color=display_props['color']),
                                    x=0.5, xanchor='center'
                                ),
                                barmode='overlay'
                            )
                            st.plotly_chart(fig_penetration, use_container_width=True)

                        st.markdown("---")
                        st.markdown("#### 🎨 Color Status Breakdown")

                        color_breakdown = loc_sku_data.groupby('On hand Inv. Color').agg({
                            'Stock': 'sum',
                            'Norm': 'sum',
                            'Location Code': 'count'
                        }).reset_index()

                        color_breakdown.columns = ['Color Status', 'Stock', 'Norm', 'Count']
                        color_breakdown['Penetration %'] = color_breakdown.apply(
                            lambda r: f"{calc_penetration(r['Norm'], r['Stock']):.2f}%", axis=1
                        )
                        color_breakdown['Stock'] = color_breakdown['Stock'].apply(lambda x: f"{int(x):,}")
                        color_breakdown['Norm'] = color_breakdown['Norm'].apply(lambda x: f"{int(x):,}")

                        st.dataframe(sanitize_for_display(color_breakdown), width='stretch', hide_index=True)

                        st.markdown("---")
                        st.markdown(f"#### 📋 Individual Locations in {loc_type}")

                        display_columns = [
                            'Location Code', 'Location Description', 'SourceLocation',
                            'Stock', 'Norm', 'Calc Penetration %', 'On hand Inv. Color',
                            'GIT/ Pending', 'Virtual Norm', 'Date'
                        ]

                        available_columns = [col for col in display_columns if col in loc_sku_data.columns]
                        display_data = loc_sku_data[available_columns].copy()

                        if 'Stock' in display_data.columns:
                            display_data['Stock'] = display_data['Stock'].apply(
                                lambda x: f"{int(x):,}" if pd.notna(x) else '0')
                        if 'Norm' in display_data.columns:
                            display_data['Norm'] = display_data['Norm'].apply(
                                lambda x: f"{int(x):,}" if pd.notna(x) else '0')
                        if 'Calc Penetration %' in display_data.columns:
                            display_data['Calc Penetration %'] = display_data['Calc Penetration %'].apply(
                                lambda x: f"{x:.2f}%" if pd.notna(x) else 'N/A')
                        if 'GIT/ Pending' in display_data.columns:
                            display_data['GIT/ Pending'] = display_data['GIT/ Pending'].apply(
                                lambda x: f"{int(x):,}" if pd.notna(x) else '0')
                        if 'Virtual Norm' in display_data.columns:
                            display_data['Virtual Norm'] = display_data['Virtual Norm'].apply(
                                lambda x: f"{int(x):,}" if pd.notna(x) and x != 0 else '-')
                        if 'Date' in display_data.columns:
                            display_data['Date'] = display_data['Date'].apply(
                                lambda x: x.strftime('%d %b %Y') if pd.notna(x) else 'N/A')

                        st.dataframe(sanitize_for_display(display_data), width='stretch', hide_index=True)
                    else:
                        st.warning(f"No data available for {loc_type}")
        else:
            st.warning("No location types found for this SKU")
    else:
        st.warning(f"No data found for SKU: {selected_sku}")

    st.markdown("---")

    # ===============================
    # SECTION 3: Complete SKU Details
    # ===============================
    st.subheader("📋 Complete Details Summary - All Locations")

    if len(sku_all_locations_data) > 0:
        st.markdown("#### 📊 Overall Summary Across All Locations")

        total_stock_all = float(sku_all_locations_data['Stock'].sum())
        total_norm_all = float(sku_all_locations_data['Norm'].sum())
        overall_pen = calc_penetration(total_norm_all, total_stock_all)

        col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns(5)
        with col_s1:
            st.metric("🏢 Total Locations", int(len(sku_all_locations_data)))
        with col_s2:
            st.metric("📦 Total Stock", f"{int(total_stock_all):,}")
        with col_s3:
            st.metric("🎯 Total Norm", f"{int(total_norm_all):,}")
        with col_s4:
            st.metric("📊 Overall Penetration", f"{overall_pen:.1f}%")
        with col_s5:
            st.metric("🔢 Location Types", int(len(sku_all_locations_data['Location Type'].unique())))

        st.markdown("---")
        st.markdown("#### 📋 Detailed Summary by Location Type")

        summary_by_location = sku_all_locations_data.groupby('Location Type').agg({
            'Stock': 'sum',
            'Norm': 'sum',
            'GIT/ Pending': 'sum',
            'Virtual Norm': 'sum',
            'Location Code': 'count'
        }).reset_index()

        summary_by_location.columns = [
            'Location Type', 'Total Stock', 'Total Norm', 'Total GIT/Pending',
            'Total Virtual Norm', 'Number of Locations'
        ]

        # Calculate penetration per location type
        summary_by_location['Penetration %'] = summary_by_location.apply(
            lambda r: f"{calc_penetration(r['Total Norm'], r['Total Stock']):.2f}%", axis=1
        )

        summary_by_location['Total Stock'] = summary_by_location['Total Stock'].apply(lambda x: f"{int(x):,}")
        summary_by_location['Total Norm'] = summary_by_location['Total Norm'].apply(lambda x: f"{int(x):,}")
        summary_by_location['Total GIT/Pending'] = summary_by_location['Total GIT/Pending'].apply(lambda x: f"{int(x):,}")
        summary_by_location['Total Virtual Norm'] = summary_by_location['Total Virtual Norm'].apply(
            lambda x: f"{int(x):,}" if x != 0 else '-')

        st.dataframe(sanitize_for_display(summary_by_location), width='stretch', hide_index=True)

        csv_all = sku_all_locations_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Complete Data for All Locations",
            data=csv_all,
            file_name=f'sku_{selected_sku}_all_locations_{date_suffix}.csv',
            mime='text/csv',
            key='download_all_locations'
        )

# ===============================
# DASHBOARD 2: BPR SUMMARY BY LOCATION & COLOR
# ===============================
elif selected_dashboard == "BPR Summary by Location & Color" and bpr_df is not None:
    st.header("📊 BPR Summary Analysis by Location Type & Color Status")

    st.sidebar.header("🔍 Filters")
    st.sidebar.subheader("📅 Date Range")

    start_date = None
    end_date = None

    if 'Date' in bpr_df.columns and bpr_df['Date'].notna().any():
        min_date = bpr_df['Date'].min()
        max_date = bpr_df['Date'].max()

        st.sidebar.caption(f"Data range: {min_date.strftime('%d %b %Y')} → {max_date.strftime('%d %b %Y')}")

        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key='summary_date_range'
        )

        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
        else:
            start_date = end_date = date_range

        df = bpr_df[
            (bpr_df['Date'].dt.normalize() >= pd.Timestamp(start_date).normalize()) &
            (bpr_df['Date'].dt.normalize() <= pd.Timestamp(end_date).normalize())
        ].copy()

        st.sidebar.info(f"📊 Total Records: {len(df):,}")
        st.sidebar.caption(f"From: {start_date.strftime('%d %b %Y')}")
        st.sidebar.caption(f"To: {end_date.strftime('%d %b %Y')}")
    else:
        df = bpr_df.copy()
        st.sidebar.warning("⚠️ No date information available")

    # Location Type filter — values from data
    st.sidebar.subheader("📍 Location Type (Optional)")
    all_locations = ['All Locations'] + sorted(df['Location Type'].dropna().unique().tolist())
    selected_location_filter = st.sidebar.selectbox("Filter by Location Type", all_locations, key='summary_location')

    if selected_location_filter != 'All Locations':
        df = df[df['Location Type'] == selected_location_filter].copy()

    st.markdown("---")

    # Colors and location types — from data
    color_order = get_all_colors(df)
    location_types = sorted(df['Location Type'].dropna().unique())

    st.subheader("📋 Summary Table: Norm & Stock by Location Type and Color Status")

    summary_data = []

    for color in color_order:
        row_data = {'Color Status': color}
        for loc_type in location_types:
            filtered = df[(df['On hand Inv. Color'] == color) & (df['Location Type'] == loc_type)]
            norm_sum = int(filtered['Norm'].sum())
            stock_sum = int(filtered['Stock'].sum())
            row_data[f'{loc_type} Norm'] = norm_sum
            row_data[f'{loc_type} Stock'] = stock_sum
            row_data[f'{loc_type} Pen%'] = f"{calc_penetration(norm_sum, stock_sum):.1f}%"
        summary_data.append(row_data)

    # Unclassified
    unclassified = df[
        df['On hand Inv. Color'].isna() |
        (df['On hand Inv. Color'].astype(str).str.strip() == '') |
        (df['On hand Inv. Color'].astype(str).str.strip() == 'Nan')
    ]
    if len(unclassified) > 0:
        row_data = {'Color Status': '(Unclassified)'}
        for loc_type in location_types:
            filtered = unclassified[unclassified['Location Type'] == loc_type]
            n = int(filtered['Norm'].sum())
            s = int(filtered['Stock'].sum())
            row_data[f'{loc_type} Norm'] = n
            row_data[f'{loc_type} Stock'] = s
            row_data[f'{loc_type} Pen%'] = f"{calc_penetration(n, s):.1f}%"
        summary_data.append(row_data)

    # Grand Total
    grand_total = {'Color Status': 'Grand Total'}
    for loc_type in location_types:
        filtered = df[df['Location Type'] == loc_type]
        n = int(filtered['Norm'].sum())
        s = int(filtered['Stock'].sum())
        grand_total[f'{loc_type} Norm'] = n
        grand_total[f'{loc_type} Stock'] = s
        grand_total[f'{loc_type} Pen%'] = f"{calc_penetration(n, s):.1f}%"
    summary_data.append(grand_total)

    summary_df = pd.DataFrame(summary_data)
    st.dataframe(sanitize_for_display(summary_df), width='stretch', hide_index=True)

    # Data Validation Check
    with st.expander("🔍 Data Validation Check", expanded=False):
        raw_total_stock = int(df['Stock'].sum())
        raw_total_norm = int(df['Norm'].sum())

        dashboard_total_stock = sum(int(grand_total.get(f'{lt} Stock', 0)) for lt in location_types)
        dashboard_total_norm = sum(int(grand_total.get(f'{lt} Norm', 0)) for lt in location_types)

        col_v1, col_v2 = st.columns(2)
        with col_v1:
            st.markdown("**Raw Data Totals:**")
            st.write(f"Total Stock: {raw_total_stock:,}")
            st.write(f"Total Norm: {raw_total_norm:,}")
            st.write(f"Penetration: {calc_penetration(raw_total_norm, raw_total_stock):.2f}%")

        with col_v2:
            st.markdown("**Dashboard Totals:**")
            st.write(f"Total Stock: {dashboard_total_stock:,}")
            st.write(f"Total Norm: {dashboard_total_norm:,}")

        if raw_total_stock == dashboard_total_stock and raw_total_norm == dashboard_total_norm:
            st.success("✅ Dashboard totals match raw data — no data loss!")
        else:
            st.error("❌ Mismatch detected!")
            if raw_total_stock != dashboard_total_stock:
                st.warning(f"Stock difference: {raw_total_stock - dashboard_total_stock:,}")
            if raw_total_norm != dashboard_total_norm:
                st.warning(f"Norm difference: {raw_total_norm - dashboard_total_norm:,}")

            missing_loc = df[df['Location Type'].isna()]
            missing_color = df[df['On hand Inv. Color'].isna() |
                               (df['On hand Inv. Color'].astype(str).str.strip() == '') |
                               (df['On hand Inv. Color'].astype(str).str.strip() == 'Nan')]
            if len(missing_loc) > 0:
                st.warning(f"⚠️ {len(missing_loc)} records have missing Location Type")
            if len(missing_color) > 0:
                st.warning(f"⚠️ {len(missing_color)} records have missing Color Status")

    # Download
    date_suffix_start = start_date.strftime("%Y%m%d") if start_date else "all"
    date_suffix_end = end_date.strftime("%Y%m%d") if end_date else "all"
    csv = summary_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Summary as CSV", data=csv,
        file_name=f'bpr_summary_{date_suffix_start}_{date_suffix_end}.csv', mime='text/csv',
    )

    st.markdown("---")

    # Visualizations
    st.subheader("📊 Visual Analysis")
    st.markdown("#### Select Location Type for Detailed Charts:")

    chart_locations = st.multiselect(
        "Select Location Types to Compare",
        options=sorted(location_types),
        default=sorted(location_types)[:min(3, len(location_types))],
        key='chart_locations'
    )

    if chart_locations:
        chart_data = []

        for color in color_order:
            for loc_type in chart_locations:
                filtered = df[(df['On hand Inv. Color'] == color) & (df['Location Type'] == loc_type)]
                chart_data.append({
                    'Color Status': color,
                    'Location Type': loc_type,
                    'Norm': int(filtered['Norm'].sum()),
                    'Stock': int(filtered['Stock'].sum())
                })

        chart_df = pd.DataFrame(chart_data)

        st.markdown("#### 📊 Norm Distribution by Color Status and Location Type")

        fig_norm = px.bar(
            chart_df, x='Color Status', y='Norm', color='Location Type',
            barmode='group', title='Norm Distribution Across Color Status and Location Types',
            height=500, color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_norm.update_layout(xaxis_title="Color Status", yaxis_title="Total Norm",
                               legend_title="Location Type", hovermode='x unified')
        st.plotly_chart(fig_norm, use_container_width=True)

        st.markdown("#### 📦 Stock Distribution by Color Status and Location Type")

        fig_stock = px.bar(
            chart_df, x='Color Status', y='Stock', color='Location Type',
            barmode='group', title='Stock Distribution Across Color Status and Location Types',
            height=500, color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_stock.update_layout(xaxis_title="Color Status", yaxis_title="Total Stock",
                                legend_title="Location Type", hovermode='x unified')
        st.plotly_chart(fig_stock, use_container_width=True)

        st.markdown("#### ⚖️ Norm vs Stock Comparison by Location Type")

        col_chart1, col_chart2 = st.columns(2)

        with col_chart1:
            loc_summary = chart_df.groupby('Location Type').agg({
                'Norm': 'sum', 'Stock': 'sum'
            }).reset_index()

            fig_comparison = go.Figure()
            fig_comparison.add_trace(go.Bar(
                name='Norm', x=loc_summary['Location Type'], y=loc_summary['Norm'],
                marker_color='#3498db', text=loc_summary['Norm'], textposition='auto'
            ))
            fig_comparison.add_trace(go.Bar(
                name='Stock', x=loc_summary['Location Type'], y=loc_summary['Stock'],
                marker_color='#e74c3c', text=loc_summary['Stock'], textposition='auto'
            ))
            fig_comparison.update_layout(
                title='Norm vs Stock by Location Type', xaxis_title="Location Type",
                yaxis_title="Quantity", barmode='group', height=400
            )
            st.plotly_chart(fig_comparison, use_container_width=True)

        with col_chart2:
            fig_pie = px.pie(
                loc_summary, values='Stock', names='Location Type',
                title='Stock Distribution by Location Type', hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_pie.update_layout(height=400)
            st.plotly_chart(fig_pie, use_container_width=True)

# ===============================
# DASHBOARD 3: SUPPLY CHAIN ANALYTICS
# ===============================
elif selected_dashboard == "Supply Chain Analytics" and supply_chain_data is not None:
    st.header("🏭 Supply Chain Analytics Dashboard")
    
    sales = supply_chain_data['sales']
    prod = supply_chain_data['prod']
    stock = supply_chain_data['stock']
    stock_btp = supply_chain_data['stock_btp']
    demand_summary = supply_chain_data['demand_summary']
    datewise_stock = supply_chain_data['datewise_stock']
    datewise_norm = supply_chain_data['datewise_norm']
    datewise_vnorm = supply_chain_data['datewise_vnorm']
    datewise_requirement = supply_chain_data['datewise_requirement']
    
    # ===============================
    # Sidebar Filters
    # ===============================
    st.sidebar.header("🔍 Filters")
    
    # SKU Selection
    sku_list = sorted(sales["SKUCode"].unique())
    selected_sku = st.sidebar.selectbox("Select SKU", sku_list, index=0, key='sc_sku')
    
    # Date Range Selection
    min_date = sales["invoice_date"].min()
    max_date = sales["invoice_date"].max()
    
    date_range = st.sidebar.date_input(
        "Select Date Range",
        [min_date, max_date],
        min_value=min_date,
        max_value=max_date,
        key='sc_date_range'
    )
    
    # ===============================
    # Filter Data
    # ===============================
    if len(date_range) == 2:
        start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    else:
        start_date = end_date = pd.to_datetime(date_range[0])
    
    # Filter Sales
    sales_f = sales[
        (sales["SKUCode"] == selected_sku) &
        (sales["invoice_date"].between(start_date, end_date))
    ]
    
    # Filter Production
    prod_f = prod[
        (prod["Matl.Code"] == selected_sku) &
        (prod["Prod.Date"].between(start_date, end_date))
    ]
    
    # Filter Stock
    stock_f = stock[
        (stock["SKUCode"] == selected_sku) &
        (stock["date"].between(start_date, end_date))
    ]
    
    # Filter BTP Stock
    stock_btp_f = stock_btp[
        (stock_btp["SKUCode"] == selected_sku) &
        (stock_btp["date"].between(start_date, end_date))
    ]
    
    # Get Demand Info
    demand_info = demand_summary[demand_summary["SKUCode"] == selected_sku]
    
    # ===============================
    # Aggregate Daily
    # ===============================
    sales_daily = sales_f.groupby("invoice_date")["volume"].sum().reset_index()
    prod_daily = prod_f.groupby("Prod.Date")["Prod.Qty."].sum().reset_index()
    stock_daily = stock_f.groupby("date")["total_qty"].mean().reset_index()
    stock_btp_daily = stock_btp_f.groupby("date")["total_qty"].mean().reset_index()
    
    # Rename Columns
    sales_daily.columns = ["Date", "Sales"]
    prod_daily.columns = ["Date", "Production"]
    stock_daily.columns = ["Date", "Stock"]
    stock_btp_daily.columns = ["Date", "BTP_Stock"]
    
    # Merge All
    df_sc = sales_daily.merge(prod_daily, on="Date", how="outer") \
                    .merge(stock_daily, on="Date", how="outer") \
                    .merge(stock_btp_daily, on="Date", how="outer") \
                    .sort_values("Date")
    
    # Fill NaN with 0 for calculations
    df_sc["Sales"] = df_sc["Sales"].fillna(0)
    df_sc["Production"] = df_sc["Production"].fillna(0)
    df_sc["Stock"] = df_sc["Stock"].fillna(0)
    df_sc["BTP_Stock"] = df_sc["BTP_Stock"].fillna(0)
    
    # Calculate additional metrics
    df_sc["Cumulative_Sales"] = df_sc["Sales"].cumsum()
    df_sc["Cumulative_Production"] = df_sc["Production"].cumsum()
    df_sc["Cumulative_Stock"] = df_sc["Stock"].cumsum()
    df_sc["Daily_Gap"] = df_sc["Production"] - df_sc["Sales"]
    df_sc["Cumulative_Gap"] = df_sc["Daily_Gap"].cumsum()
    
    # Calculate normalized stocks
    if df_sc["Stock"].max() > 0:
        df_sc["Norm_Stock"] = (df_sc["Stock"] / df_sc["Stock"].max()) * 100
    else:
        df_sc["Norm_Stock"] = 0
        
    if df_sc["BTP_Stock"].max() > 0:
        df_sc["Norm_BTP_Stock"] = (df_sc["BTP_Stock"] / df_sc["BTP_Stock"].max()) * 100
    else:
        df_sc["Norm_BTP_Stock"] = 0
    
    # ===============================
    # SKU Information
    # ===============================
    st.markdown("### 📦 SKU Information")
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        st.info(f"**Selected SKU:** {selected_sku}")
        if not sales_f.empty:
            sku_desc = sales_f["Description"].iloc[0]
            st.info(f"**Description:** {sku_desc}")
    
    with col_info2:
        st.info(f"**Date Range:** {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        st.info(f"**Data Points:** {len(df_sc)} days")
    
    # ===============================
    # KPI Metrics
    # ===============================
    st.markdown("### 📈 Key Performance Indicators")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total_sales = int(df_sc["Sales"].sum())
    total_production = int(df_sc["Production"].sum())
    avg_stock = int(df_sc["Stock"].mean())
    demand_supply_gap = total_sales - total_production
    current_stock = int(df_sc["Stock"].iloc[-1]) if not df_sc.empty and df_sc["Stock"].iloc[-1] > 0 else 0
    
    col1.metric("📦 Total Sales", f"{total_sales:,}", 
                delta=None, 
                help="Total units sold in selected period")
    
    col2.metric("🏭 Total Production", f"{total_production:,}", 
                delta=f"{demand_supply_gap:+,}" if demand_supply_gap != 0 else "0",
                delta_color="inverse",
                help="Total units produced in selected period")
    
    col3.metric("📊 Avg Daily Stock", f"{avg_stock:,}", 
                help="Average inventory level")
    
    col4.metric("⚠️ Demand-Supply Gap", f"{abs(demand_supply_gap):,}", 
                delta="Shortage" if demand_supply_gap > 0 else "Surplus",
                delta_color="inverse" if demand_supply_gap > 0 else "normal",
                help="Difference between sales and production")
    
    col5.metric("📦 Current Stock", f"{current_stock:,}", 
                help="Latest stock level")
    
    # ===============================
    # Main Charts - Sales vs Production vs Stock
    # ===============================
    st.markdown("### 📊 Sales vs Production vs Stock Trend")
    
    # ===============================
    # Market Segment Data Options
    # ===============================
    st.markdown("#### 📦 Include Market Segment Data (from Datewise Comparison)")
    
    # Filter datewise for current SKU
    sku_markets_stock = datewise_stock[datewise_stock["SKUCode"] == selected_sku].copy()
    sku_markets_norm = datewise_norm[datewise_norm["SKUCode"] == selected_sku].copy()
    sku_markets_vnorm = datewise_vnorm[datewise_vnorm["SKUCode"] == selected_sku].copy()
    sku_markets_requirement = datewise_requirement[datewise_requirement["SKUCode"] == selected_sku].copy()
    
    has_market_data = not sku_markets_stock.empty
    
    if has_market_data:
        # Get available market segments
        available_markets = sorted(sku_markets_stock['Market'].unique())
        
        # Market segment selector
        col_ms1, col_ms2, col_ms3 = st.columns(3)
        
        main_selected_markets = []
        with col_ms1:
            if 'OE' in available_markets:
                if st.checkbox("Include OE (Original Equipment)", value=False, key="main_oe_check"):
                    main_selected_markets.append('OE')
        
        with col_ms2:
            if 'RE' in available_markets:
                if st.checkbox("Include RE (Replacement)", value=False, key="main_re_check"):
                    main_selected_markets.append('RE')
        
        with col_ms3:
            if 'EXP' in available_markets:
                if st.checkbox("Include EXP (Export)", value=False, key="main_exp_check"):
                    main_selected_markets.append('EXP')
        
        # Data type selector
        if main_selected_markets:
            st.markdown("**Select Data Types to Display:**")
            col_dt1, col_dt2, col_dt3, col_dt4 = st.columns(4)
            
            with col_dt1:
                show_market_stock = st.checkbox("Market Stock", value=True, key="main_show_stock")
            with col_dt2:
                show_market_requirement = st.checkbox("Market Requirement", value=False, key="main_show_req")
            with col_dt3:
                show_market_norm = st.checkbox("Market Norm", value=False, key="main_show_norm")
            with col_dt4:
                show_market_vnorm = st.checkbox("Market Virtual Norm", value=False, key="main_show_vnorm")
    else:
        main_selected_markets = []
    
    st.markdown("---")
    
    # ===============================
    # Build Combined Chart
    # ===============================
    fig1 = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Daily Sales, Production & Stock", "Cumulative Sales, Production & Stock"),
        vertical_spacing=0.12,
        row_heights=[0.5, 0.5],
        specs=[[{"secondary_y": True}], [{"secondary_y": True}]]
    )
    
    # Daily Sales and Production (primary y-axis)
    fig1.add_trace(
        go.Scatter(x=df_sc["Date"], y=df_sc["Sales"], 
                   name="Sales", mode='lines+markers',
                   line=dict(color='#ff7f0e', width=2),
                   marker=dict(size=6)),
        row=1, col=1, secondary_y=False
    )
    
    fig1.add_trace(
        go.Scatter(x=df_sc["Date"], y=df_sc["Production"], 
                   name="Production", mode='lines+markers',
                   line=dict(color='#2ca02c', width=2),
                   marker=dict(size=6)),
        row=1, col=1, secondary_y=False
    )
    
    # Daily Stock (secondary y-axis)
    fig1.add_trace(
        go.Scatter(x=df_sc["Date"], y=df_sc["Stock"], 
                   name="Stock", mode='lines+markers',
                   line=dict(color='#1f77b4', width=2),
                   marker=dict(size=6)),
        row=1, col=1, secondary_y=True
    )
    
    # BTP Stock if available (secondary y-axis)
    if df_sc["BTP_Stock"].sum() > 0:
        fig1.add_trace(
            go.Scatter(x=df_sc["Date"], y=df_sc["BTP_Stock"], 
                       name="BTP Stock", mode='lines+markers',
                       line=dict(color='#9467bd', width=2, dash='dot'),
                       marker=dict(size=6, symbol='diamond')),
            row=1, col=1, secondary_y=True
        )
    
    # Add Market Segment Data if selected
    if has_market_data and main_selected_markets:
        date_columns = [col for col in datewise_stock.columns if col not in ["SKU+Market", "SKUCode", "SKU Description", "Market"]]
        
        # VIBRANT, DISTINCT COLORS FOR EACH MARKET AND DATA TYPE
        data_type_colors = {
            'stock': {
                'OE': '#FF1744',      # Vibrant Red
                'RE': '#00B0FF',      # Bright Cyan Blue
                'EXP': '#00E676'      # Bright Green
            },
            'requirement': {
                'OE': '#FF6E40',      # Deep Orange
                'RE': '#448AFF',      # Indigo Blue
                'EXP': '#69F0AE'      # Light Green
            },
            'norm': {
                'OE': '#F50057',      # Pink Red
                'RE': '#2979FF',      # Royal Blue
                'EXP': '#00C853'      # Medium Green
            },
            'vnorm': {
                'OE': '#D500F9',      # Purple
                'RE': '#00E5FF',      # Cyan
                'EXP': '#AEEA00'      # Lime
            }
        }
        
        line_styles = {
            'stock': 'solid',
            'requirement': 'dash',
            'norm': 'dot',
            'vnorm': 'dashdot'
        }
        
        line_widths = {
            'stock': 3,
            'requirement': 3,
            'norm': 2.5,
            'vnorm': 2.5
        }
        
        for market in main_selected_markets:
            # Market Stock
            if show_market_stock:
                market_data = sku_markets_stock[sku_markets_stock['Market'] == market]
                if not market_data.empty:
                    values = market_data[date_columns].values[0]
                    ts_market = pd.DataFrame({
                        'Date': [pd.to_datetime(col, format='%d%m%Y') for col in date_columns],
                        'Value': values
                    })
                    ts_market = ts_market[
                        (ts_market['Date'] >= start_date) & 
                        (ts_market['Date'] <= end_date)
                    ].sort_values('Date').dropna()
                    
                    if not ts_market.empty:
                        fig1.add_trace(
                            go.Scatter(x=ts_market["Date"], y=ts_market["Value"],
                                       name=f"{market} Stock",
                                       mode='lines+markers',
                                       line=dict(color=data_type_colors['stock'][market], 
                                               width=line_widths['stock'], 
                                               dash=line_styles['stock']),
                                       marker=dict(size=7, color=data_type_colors['stock'][market])),
                            row=1, col=1, secondary_y=True
                        )
            
            # Market Requirement
            if show_market_requirement:
                market_req = sku_markets_requirement[sku_markets_requirement['Market'] == market]
                if not market_req.empty:
                    values = market_req[date_columns].values[0]
                    ts_req = pd.DataFrame({
                        'Date': [pd.to_datetime(col, format='%d%m%Y') for col in date_columns],
                        'Value': values
                    })
                    ts_req = ts_req[
                        (ts_req['Date'] >= start_date) & 
                        (ts_req['Date'] <= end_date)
                    ].sort_values('Date').dropna()
                    
                    if not ts_req.empty:
                        fig1.add_trace(
                            go.Scatter(x=ts_req["Date"], y=ts_req["Value"],
                                       name=f"{market} Requirement",
                                       mode='lines+markers',
                                       line=dict(color=data_type_colors['requirement'][market], 
                                               width=line_widths['requirement'], 
                                               dash=line_styles['requirement']),
                                       marker=dict(size=7, color=data_type_colors['requirement'][market], symbol='square')),
                            row=1, col=1, secondary_y=True
                        )
            
            # Market Norm
            if show_market_norm:
                market_norm = sku_markets_norm[sku_markets_norm['Market'] == market]
                if not market_norm.empty:
                    values = market_norm[date_columns].values[0]
                    ts_norm = pd.DataFrame({
                        'Date': [pd.to_datetime(col, format='%d%m%Y') for col in date_columns],
                        'Value': values
                    })
                    ts_norm = ts_norm[
                        (ts_norm['Date'] >= start_date) & 
                        (ts_norm['Date'] <= end_date)
                    ].sort_values('Date').dropna()
                    
                    if not ts_norm.empty:
                        fig1.add_trace(
                            go.Scatter(x=ts_norm["Date"], y=ts_norm["Value"],
                                       name=f"{market} Norm",
                                       mode='lines+markers',
                                       line=dict(color=data_type_colors['norm'][market], 
                                               width=line_widths['norm'], 
                                               dash=line_styles['norm']),
                                       marker=dict(size=6, color=data_type_colors['norm'][market], symbol='diamond')),
                            row=1, col=1, secondary_y=True
                        )
            
            # Market Virtual Norm
            if show_market_vnorm:
                market_vnorm = sku_markets_vnorm[sku_markets_vnorm['Market'] == market]
                if not market_vnorm.empty:
                    values = market_vnorm[date_columns].values[0]
                    ts_vnorm = pd.DataFrame({
                        'Date': [pd.to_datetime(col, format='%d%m%Y') for col in date_columns],
                        'Value': values
                    })
                    ts_vnorm = ts_vnorm[
                        (ts_vnorm['Date'] >= start_date) & 
                        (ts_vnorm['Date'] <= end_date)
                    ].sort_values('Date').dropna()
                    
                    if not ts_vnorm.empty:
                        fig1.add_trace(
                            go.Scatter(x=ts_vnorm["Date"], y=ts_vnorm["Value"],
                                       name=f"{market} Virtual Norm",
                                       mode='lines+markers',
                                       line=dict(color=data_type_colors['vnorm'][market], 
                                               width=line_widths['vnorm'], 
                                               dash=line_styles['vnorm']),
                                       marker=dict(size=6, color=data_type_colors['vnorm'][market], symbol='cross')),
                            row=1, col=1, secondary_y=True
                        )
    
    # Cumulative Sales and Production (primary y-axis)
    fig1.add_trace(
        go.Scatter(x=df_sc["Date"], y=df_sc["Cumulative_Sales"], 
                   name="Cumulative Sales", mode='lines',
                   line=dict(color='#ff7f0e', width=3, dash='dash'),
                   fill='tozeroy', fillcolor='rgba(255, 127, 14, 0.1)'),
        row=2, col=1, secondary_y=False
    )
    
    fig1.add_trace(
        go.Scatter(x=df_sc["Date"], y=df_sc["Cumulative_Production"], 
                   name="Cumulative Production", mode='lines',
                   line=dict(color='#2ca02c', width=3, dash='dash'),
                   fill='tozeroy', fillcolor='rgba(44, 160, 44, 0.1)'),
        row=2, col=1, secondary_y=False
    )
    
    # Cumulative Stock (secondary y-axis)
    df_sc["Cumulative_Stock"] = df_sc["Stock"].cumsum()
    fig1.add_trace(
        go.Scatter(x=df_sc["Date"], y=df_sc["Cumulative_Stock"], 
                   name="Cumulative Stock", mode='lines',
                   line=dict(color='#1f77b4', width=3, dash='dash'),
                   fill='tozeroy', fillcolor='rgba(31, 119, 180, 0.1)'),
        row=2, col=1, secondary_y=True
    )
    
    # Update axes
    fig1.update_xaxes(title_text="Date", row=1, col=1)
    fig1.update_xaxes(title_text="Date", row=2, col=1)
    fig1.update_yaxes(title_text="Sales & Production Quantity", row=1, col=1, secondary_y=False)
    fig1.update_yaxes(title_text="Stock Level", row=1, col=1, secondary_y=True)
    fig1.update_yaxes(title_text="Cumulative Sales & Production", row=2, col=1, secondary_y=False)
    fig1.update_yaxes(title_text="Cumulative Stock", row=2, col=1, secondary_y=True)
    
    fig1.update_layout(height=700, showlegend=True, hovermode='x unified')
    
    st.plotly_chart(fig1, use_container_width=True)
    
    # ===============================
    # Stock + BTP Combined
    # ===============================
    st.markdown("### 📦 Stock Level Trend (Regular + BTP)")
    
    fig2 = go.Figure()
    
    # Regular Stock
    fig2.add_trace(
        go.Scatter(x=df_sc["Date"], y=df_sc["Stock"], 
                   name="Regular Stock", mode='lines+markers',
                   line=dict(color='#1f77b4', width=3),
                   marker=dict(size=7),
                   fill='tozeroy', fillcolor='rgba(31, 119, 180, 0.2)')
    )
    
    # BTP Stock (different color)
    if df_sc["BTP_Stock"].sum() > 0:
        fig2.add_trace(
            go.Scatter(x=df_sc["Date"], y=df_sc["BTP_Stock"], 
                       name="BTP Stock", mode='lines+markers',
                       line=dict(color='#9467bd', width=3, dash='dot'),
                       marker=dict(size=7, symbol='diamond'),
                       fill='tozeroy', fillcolor='rgba(148, 103, 189, 0.2)')
        )
    
    # Add average lines
    if df_sc["Stock"].sum() > 0:
        avg_line = df_sc["Stock"].mean()
        fig2.add_hline(y=avg_line, line_dash="dash", line_color="#1f77b4",
                       annotation_text=f"Avg Regular: {avg_line:.0f}",
                       annotation_position="left")
    
    if df_sc["BTP_Stock"].sum() > 0:
        avg_btp_line = df_sc["BTP_Stock"].mean()
        fig2.add_hline(y=avg_btp_line, line_dash="dash", line_color="#9467bd",
                       annotation_text=f"Avg BTP: {avg_btp_line:.0f}",
                       annotation_position="right")
    
    fig2.update_layout(
        xaxis_title="Date",
        yaxis_title="Stock Quantity",
        height=500,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig2, use_container_width=True)
    
    # ===============================
    # Market Segment Stock Analysis from XLSX
    # ===============================
    st.markdown("---")
    st.markdown("### 📊 Market Segment Stock Analysis (from Datewise Comparison)")
    
    if not sku_markets_stock.empty:
        # Get SKU Description
        sku_description = sku_markets_stock['SKU Description'].iloc[0] if 'SKU Description' in sku_markets_stock.columns else "N/A"
        
        # Display SKU Info
        st.markdown(f"**SKU Code:** {selected_sku}")
        st.markdown(f"**Description:** {sku_description}")
        st.markdown("---")
        
        # Get available market segments for this SKU
        available_markets = sorted(sku_markets_stock['Market'].unique())
        
        # Market segment selector
        st.markdown("#### Select Market Segments:")
        col_m1, col_m2, col_m3 = st.columns(3)
        
        selected_markets = []
        with col_m1:
            if 'OE' in available_markets:
                if st.checkbox("OE (Original Equipment)", value=True, key="oe_check"):
                    selected_markets.append('OE')
        
        with col_m2:
            if 'RE' in available_markets:
                if st.checkbox("RE (Replacement)", value=True, key="re_check"):
                    selected_markets.append('RE')
        
        with col_m3:
            if 'EXP' in available_markets:
                if st.checkbox("EXP (Export)", value=True, key="exp_check"):
                    selected_markets.append('EXP')
        
        if selected_markets:
            # Prepare data for plotting
            date_columns = [col for col in datewise_stock.columns if col not in ["SKU+Market", "SKUCode", "SKU Description", "Market"]]
            
            # ===============================
            # Visualization Options
            # ===============================
            st.markdown("---")
            st.markdown("#### 📊 Select Visualizations to Display:")
            
            col_v1, col_v2, col_v3, col_v4, col_v5 = st.columns(5)
            
            with col_v1:
                show_stock = st.checkbox("📈 Stock Levels", value=True, key="show_stock")
            
            with col_v2:
                show_requirement = st.checkbox("📋 Requirement", value=False, key="show_requirement")
            
            with col_v3:
                show_cumulative = st.checkbox("📊 Cumulative Stock", value=True, key="show_cumulative")
            
            with col_v4:
                show_norm = st.checkbox("📉 Norm Data", value=False, key="show_norm")
            
            with col_v5:
                show_virtual_norm = st.checkbox("🔄 Virtual Norm Data", value=False, key="show_virtual_norm")
            
            # VIBRANT, DISTINCT COLORS FOR EACH MARKET
            colors_market = {
                'OE': '#FF1744',      # Vibrant Red
                'RE': '#00B0FF',      # Bright Cyan Blue
                'EXP': '#00E676'      # Bright Green
            }
            
            market_data_dict = {}  # Store stock data
            market_norm_dict = {}  # Store norm data
            market_vnorm_dict = {}  # Store virtual norm data
            market_requirement_dict = {}  # Store requirement data
            
            # Process market data
            for market in selected_markets:
                # Get stock data
                market_data = sku_markets_stock[sku_markets_stock['Market'] == market]
                
                if not market_data.empty:
                    # Extract stock values
                    values = market_data[date_columns].values[0]
                    
                    # Create DataFrame for stock
                    ts_market = pd.DataFrame({
                        'Date': [pd.to_datetime(col, format='%d%m%Y') for col in date_columns],
                        'Stock': values
                    })
                    
                    # Filter by selected date range
                    ts_market = ts_market[
                        (ts_market['Date'] >= start_date) & 
                        (ts_market['Date'] <= end_date)
                    ].sort_values('Date').dropna()
                    
                    if not ts_market.empty:
                        market_data_dict[market] = ts_market.copy()
                
                # Get norm data
                market_norm = sku_markets_norm[sku_markets_norm['Market'] == market]
                
                if not market_norm.empty:
                    norm_values = market_norm[date_columns].values[0]
                    ts_norm = pd.DataFrame({
                        'Date': [pd.to_datetime(col, format='%d%m%Y') for col in date_columns],
                        'Norm': norm_values
                    })
                    ts_norm = ts_norm[
                        (ts_norm['Date'] >= start_date) & 
                        (ts_norm['Date'] <= end_date)
                    ].sort_values('Date').dropna()
                    
                    if not ts_norm.empty:
                        market_norm_dict[market] = ts_norm.copy()
                
                # Get virtual norm data
                market_vnorm = sku_markets_vnorm[sku_markets_vnorm['Market'] == market]
                
                if not market_vnorm.empty:
                    vnorm_values = market_vnorm[date_columns].values[0]
                    ts_vnorm = pd.DataFrame({
                        'Date': [pd.to_datetime(col, format='%d%m%Y') for col in date_columns],
                        'VirtualNorm': vnorm_values
                    })
                    ts_vnorm = ts_vnorm[
                        (ts_vnorm['Date'] >= start_date) & 
                        (ts_vnorm['Date'] <= end_date)
                    ].sort_values('Date').dropna()
                    
                    if not ts_vnorm.empty:
                        market_vnorm_dict[market] = ts_vnorm.copy()
                
                # Get requirement data
                market_req = sku_markets_requirement[sku_markets_requirement['Market'] == market]
                
                if not market_req.empty:
                    req_values = market_req[date_columns].values[0]
                    ts_req = pd.DataFrame({
                        'Date': [pd.to_datetime(col, format='%d%m%Y') for col in date_columns],
                        'Requirement': req_values
                    })
                    ts_req = ts_req[
                        (ts_req['Date'] >= start_date) & 
                        (ts_req['Date'] <= end_date)
                    ].sort_values('Date').dropna()
                    
                    if not ts_req.empty:
                        market_requirement_dict[market] = ts_req.copy()
            
            # ===============================
            # 1. Stock Levels by Market
            # ===============================
            if show_stock and market_data_dict:
                st.markdown("#### 📈 Stock Levels by Market Segment")
                
                fig_market_stock = go.Figure()
                
                for market in selected_markets:
                    if market in market_data_dict:
                        ts_market = market_data_dict[market]
                        
                        # Plot stock levels with vibrant colors
                        fig_market_stock.add_trace(
                            go.Scatter(
                                x=ts_market["Date"],
                                y=ts_market["Stock"],
                                name=f"{market} Stock",
                                mode='lines+markers',
                                line=dict(color=colors_market.get(market, '#95a5a6'), width=4),
                                marker=dict(size=9, color=colors_market.get(market, '#95a5a6'))
                            )
                        )
                
                fig_market_stock.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Stock Quantity",
                    height=500,
                    hovermode='x unified',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                st.plotly_chart(fig_market_stock, use_container_width=True)
            
            # ===============================
            # 2. Requirement by Market
            # ===============================
            if show_requirement:
                if market_requirement_dict:
                    st.markdown("#### 📋 Requirement by Market Segment")
                    st.markdown("*Displaying requirement data from the 'Requirement' sheet*")
                    
                    fig_market_req = go.Figure()
                    
                    # Different shades for requirements
                    req_colors = {
                        'OE': '#FF6E40',      # Deep Orange
                        'RE': '#448AFF',      # Indigo Blue
                        'EXP': '#69F0AE'      # Light Green
                    }
                    
                    for market in selected_markets:
                        if market in market_requirement_dict:
                            ts_req = market_requirement_dict[market]
                            
                            fig_market_req.add_trace(
                                go.Scatter(
                                    x=ts_req["Date"],
                                    y=ts_req["Requirement"],
                                    name=f"{market} Requirement",
                                    mode='lines+markers',
                                    line=dict(color=req_colors.get(market, '#95a5a6'), width=4),
                                    marker=dict(size=9, color=req_colors.get(market, '#95a5a6'), symbol='square')
                                )
                            )
                    
                    fig_market_req.update_layout(
                        xaxis_title="Date",
                        yaxis_title="Requirement Quantity",
                        height=500,
                        hovermode='x unified',
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    
                    st.plotly_chart(fig_market_req, use_container_width=True)
                else:
                    st.info("No requirement data found for selected markets in the selected date range.")
            
            # ===============================
            # 3. Cumulative Stock by Market
            # ===============================
            if show_cumulative and market_data_dict:
                st.markdown("#### 📊 Cumulative Stock by Market Segment")
                
                fig_market_cumulative = go.Figure()
                
                for market in selected_markets:
                    if market in market_data_dict:
                        ts_market = market_data_dict[market].copy()
                        ts_market['Cumulative'] = ts_market['Stock'].cumsum()
                        
                        fig_market_cumulative.add_trace(
                            go.Scatter(
                                x=ts_market["Date"],
                                y=ts_market["Cumulative"],
                                name=f"{market} Cumulative",
                                mode='lines+markers',
                                line=dict(color=colors_market.get(market, '#95a5a6'), width=4),
                                marker=dict(size=9, color=colors_market.get(market, '#95a5a6')),
                                fill='tozeroy',
                                fillcolor=f'rgba{tuple(list(int(colors_market.get(market, "#95a5a6")[i:i+2], 16) for i in (1, 3, 5)) + [0.2])}'
                            )
                        )
                
                fig_market_cumulative.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Cumulative Stock",
                    height=500,
                    hovermode='x unified',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                st.plotly_chart(fig_market_cumulative, use_container_width=True)
            
            # ===============================
            # 4. Norm Data by Market
            # ===============================
            if show_norm:
                if market_norm_dict:
                    st.markdown("#### 📉 Norm Data by Market Segment")
                    st.markdown("*Displaying norm data from the 'Norm' sheet*")
                    
                    fig_market_norm = go.Figure()
                    
                    # Different colors for norm
                    norm_colors = {
                        'OE': '#F50057',      # Pink Red
                        'RE': '#2979FF',      # Royal Blue
                        'EXP': '#00C853'      # Medium Green
                    }
                    
                    for market in selected_markets:
                        if market in market_norm_dict:
                            ts_norm = market_norm_dict[market]
                            
                            fig_market_norm.add_trace(
                                go.Scatter(
                                    x=ts_norm["Date"],
                                    y=ts_norm["Norm"],
                                    name=f"{market} Norm",
                                    mode='lines+markers',
                                    line=dict(color=norm_colors.get(market, '#95a5a6'), width=3),
                                    marker=dict(size=8, color=norm_colors.get(market, '#95a5a6'), symbol='diamond')
                                )
                            )
                    
                    fig_market_norm.update_layout(
                        xaxis_title="Date",
                        yaxis_title="Norm Value",
                        height=450,
                        hovermode='x unified',
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    
                    st.plotly_chart(fig_market_norm, use_container_width=True)
                else:
                    st.info("No norm data found for selected markets in the selected date range.")
            
            # ===============================
            # 5. Virtual Norm Data by Market
            # ===============================
            if show_virtual_norm:
                if market_vnorm_dict:
                    st.markdown("#### 🔄 Virtual Norm Data by Market Segment")
                    st.markdown("*Displaying virtual norm data from the 'Virtual Norm' sheet*")
                    
                    fig_virtual_norm = go.Figure()
                    
                    # Different colors for virtual norm
                    vnorm_colors = {
                        'OE': '#D500F9',      # Purple
                        'RE': '#00E5FF',      # Cyan
                        'EXP': '#AEEA00'      # Lime
                    }
                    
                    for market in selected_markets:
                        if market in market_vnorm_dict:
                            ts_vnorm = market_vnorm_dict[market]
                            
                            fig_virtual_norm.add_trace(
                                go.Scatter(
                                    x=ts_vnorm["Date"],
                                    y=ts_vnorm["VirtualNorm"],
                                    name=f"{market} Virtual Norm",
                                    mode='lines+markers',
                                    line=dict(color=vnorm_colors.get(market, '#95a5a6'), width=3),
                                    marker=dict(size=8, color=vnorm_colors.get(market, '#95a5a6'), symbol='cross')
                                )
                            )
                    
                    fig_virtual_norm.update_layout(
                        xaxis_title="Date",
                        yaxis_title="Virtual Norm Value",
                        height=450,
                        hovermode='x unified',
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    
                    st.plotly_chart(fig_virtual_norm, use_container_width=True)
                else:
                    st.info("No virtual norm data found for selected markets in the selected date range.")
            
            # ===============================
            # Market Comparison Table
            # ===============================
            st.markdown("---")
            st.markdown("#### 📋 Market Segment Summary Table")
            
            summary_data = []
            
            for market in selected_markets:
                row_data = {'Market': market}
                
                # Stock data
                if market in market_data_dict:
                    ts_market = market_data_dict[market].copy()
                    row_data.update({
                        'Total Stock': int(ts_market['Stock'].sum()),
                        'Average Daily Stock': int(ts_market['Stock'].mean()),
                        'Max Daily Stock': int(ts_market['Stock'].max()),
                        'Min Daily Stock': int(ts_market['Stock'].min()),
                        'Current Stock': int(ts_market['Stock'].iloc[-1]),
                    })
                
                # Requirement data
                if market in market_requirement_dict:
                    ts_req = market_requirement_dict[market]
                    row_data.update({
                        'Total Requirement': int(ts_req['Requirement'].sum()),
                        'Avg Daily Requirement': int(ts_req['Requirement'].mean()),
                        'Current Requirement': int(ts_req['Requirement'].iloc[-1])
                    })
                
                # Norm data
                if market in market_norm_dict:
                    ts_norm = market_norm_dict[market]
                    row_data.update({
                        'Avg Norm': f"{ts_norm['Norm'].mean():.2f}",
                        'Current Norm': f"{ts_norm['Norm'].iloc[-1]:.2f}"
                    })
                
                # Virtual norm data
                if market in market_vnorm_dict:
                    ts_vnorm = market_vnorm_dict[market]
                    row_data.update({
                        'Avg Virtual Norm': f"{ts_vnorm['VirtualNorm'].mean():.2f}",
                        'Current Virtual Norm': f"{ts_vnorm['VirtualNorm'].iloc[-1]:.2f}"
                    })
                
                row_data['Days with Data'] = len(market_data_dict.get(market, []))
                summary_data.append(row_data)
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
        else:
            st.info("Please select at least one market segment to view the analysis.")
    else:
        st.warning(f"No market segment data available for SKU: {selected_sku}")
    
    # ===============================
    # Combined View
    # ===============================
    st.markdown("---")
    st.markdown("### 🔄 Integrated View: Sales, Production & Stock")
    
    fig5 = go.Figure()
    
    fig5.add_trace(
        go.Scatter(x=df_sc["Date"], y=df_sc["Sales"], 
                   name="Sales", mode='lines+markers',
                   line=dict(color='#ff7f0e', width=2),
                   yaxis='y1')
    )
    
    fig5.add_trace(
        go.Scatter(x=df_sc["Date"], y=df_sc["Production"], 
                   name="Production", mode='lines+markers',
                   line=dict(color='#2ca02c', width=2),
                   yaxis='y1')
    )
    
    fig5.add_trace(
        go.Scatter(x=df_sc["Date"], y=df_sc["Stock"], 
                   name="Regular Stock", mode='lines',
                   line=dict(color='#1f77b4', width=2),
                   yaxis='y2')
    )
    
    if df_sc["BTP_Stock"].sum() > 0:
        fig5.add_trace(
            go.Scatter(x=df_sc["Date"], y=df_sc["BTP_Stock"], 
                       name="BTP Stock", mode='lines',
                       line=dict(color='#9467bd', width=2, dash='dot'),
                       yaxis='y2')
        )
    
    fig5.update_layout(
        xaxis=dict(title="Date"),
        yaxis=dict(title="Sales & Production Quantity", side='left'),
        yaxis2=dict(title="Stock Level", side='right', overlaying='y'),
        height=500,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig5, use_container_width=True)
    
    # ===============================
    # Data Table
    # ===============================
    with st.expander("📋 View Detailed Data"):
        display_cols = ['Date', 'Sales', 'Production', 'Stock', 'BTP_Stock', 'Daily_Gap', 'Cumulative_Sales', 'Cumulative_Production', 'Cumulative_Stock', 'Cumulative_Gap', 'Norm_Stock', 'Norm_BTP_Stock']
        display_df = df_sc[display_cols].copy()
        
        st.dataframe(
            display_df.style.format({
                'Sales': '{:.0f}',
                'Production': '{:.0f}',
                'Stock': '{:.0f}',
                'BTP_Stock': '{:.0f}',
                'Daily_Gap': '{:+.0f}',
                'Cumulative_Sales': '{:.0f}',
                'Cumulative_Production': '{:.0f}',
                'Cumulative_Stock': '{:.0f}',
                'Cumulative_Gap': '{:+.0f}',
                'Norm_Stock': '{:.1f}%',
                'Norm_BTP_Stock': '{:.1f}%'
            }),
            use_container_width=True
        )
        
        # Download button
        csv = df_sc.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Data as CSV",
            data=csv,
            file_name=f'supply_chain_data_{selected_sku}_{start_date.strftime("%Y%m%d")}_{end_date.strftime("%Y%m%d")}.csv',
            mime='text/csv',
        )
    
    # ===============================
    # Summary Statistics
    # ===============================
    with st.expander("📊 Summary Statistics"):
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        
        with col_stat1:
            st.markdown("#### Sales Statistics")
            st.write(f"**Mean Daily Sales:** {df_sc['Sales'].mean():.2f}")
            st.write(f"**Max Daily Sales:** {df_sc['Sales'].max():.0f}")
            st.write(f"**Min Daily Sales:** {df_sc['Sales'].min():.0f}")
            st.write(f"**Std Dev:** {df_sc['Sales'].std():.2f}")
        
        with col_stat2:
            st.markdown("#### Production Statistics")
            st.write(f"**Mean Daily Production:** {df_sc['Production'].mean():.2f}")
            st.write(f"**Max Daily Production:** {df_sc['Production'].max():.0f}")
            st.write(f"**Min Daily Production:** {df_sc['Production'].min():.0f}")
            st.write(f"**Std Dev:** {df_sc['Production'].std():.2f}")
        
        with col_stat3:
            st.markdown("#### Regular Stock Statistics")
            st.write(f"**Mean Stock Level:** {df_sc['Stock'].mean():.2f}")
            st.write(f"**Max Stock Level:** {df_sc['Stock'].max():.0f}")
            st.write(f"**Min Stock Level:** {df_sc['Stock'].min():.0f}")
            st.write(f"**Std Dev:** {df_sc['Stock'].std():.2f}")
        
        with col_stat4:
            st.markdown("#### BTP Stock Statistics")
            st.write(f"**Mean BTP Stock:** {df_sc['BTP_Stock'].mean():.2f}")
            st.write(f"**Max BTP Stock:** {df_sc['BTP_Stock'].max():.0f}")
            st.write(f"**Min BTP Stock:** {df_sc['BTP_Stock'].min():.0f}")
            st.write(f"**Std Dev:** {df_sc['BTP_Stock'].std():.2f}")

# ===============================
# Footer
# ===============================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #7f8c8d; padding: 1rem;'>
        <p>📊 Integrated Supply Chain & BPR Analytics Dashboard | Data-Driven Insights for Better Decision Making</p>
    </div>
    """,
    unsafe_allow_html=True
)
