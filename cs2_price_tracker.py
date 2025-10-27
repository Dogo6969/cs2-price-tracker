# cs2_price_tracker.py
import os
import requests
import pandas as pd
from datetime import datetime, timedelta
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from urllib.parse import urlencode
import base64
from io import BytesIO
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from prophet import Prophet
import requests
import time
# ================== REPORT LAB ==================
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.fonts import addMapping
from reportlab.platypus import Table
from reportlab.pdfbase.pdfmetrics import stringWidth

st.set_page_config(page_title="PDF Dashboard Skin Steam", layout="wide")
API_KEY = st.secrets["API_KEY"]   # đăng ký với dịch vụ API
API_URL = "https://api.csgoskins.gg/api/v1/basic-item-details"
# ================== DEJAVUSans ==================
pdfmetrics.registerFont(TTFont('DejaVuSans', 'fonts/DejaVuSans.ttf'))
pdfmetrics.registerFont(TTFont('DejaVuSans-Bold', 'fonts/DejaVuSans-Bold.ttf'))
pdfmetrics.registerFont(TTFont('DejaVuSans-Oblique', 'fonts/DejaVuSans-Oblique.ttf'))
pdfmetrics.registerFont(TTFont('DejaVuSans-BoldOblique', 'fonts/DejaVuSans-BoldOblique.ttf'))
addMapping('DejaVuSans', 0, 0, 'DejaVuSans')             # normal
addMapping('DejaVuSans', 0, 1, 'DejaVuSans-Oblique')     # italic
addMapping('DejaVuSans', 1, 0, 'DejaVuSans-Bold')        # bold
addMapping('DejaVuSans', 1, 1, 'DejaVuSans-BoldOblique') # bold+italic
# ================== DANH SÁCH SKIN ==================
skins = [
    "AWP | Neo-Noir (Minimal Wear)",
    "M4A4 | Neo-Noir (Field-Tested)",
    "Glock-18 | Neo-Noir (Field-Tested)",
    "USP-S | Neo-Noir (Field-Tested)",
    "Desert Eagle | Printstream (Field-Tested)",

    "★ Gut Knife | Gamma Doppler (Factory New)",
    "★ Shadow Daggers | Marble Fade (Factory New)",
    "★ Driver Gloves | Crimson Weave (Field-Tested)",
    "★ Hand Wraps | Slaughter (Field-Tested)",
    "★ Falchion Knife | Doppler (Factory New)"
]

target_prices = {
    "AWP | Neo-Noir (Minimal Wear)": 1200000,
    "M4A4 | Neo-Noir (Field-Tested)": 900000,
    "Glock-18 | Neo-Noir (Field-Tested)": 200000,
    "USP-S | Neo-Noir (Field-Tested)": 1500000,
    "Desert Eagle | Printstream (Field-Tested)": 1000000,

    "★ Gut Knife | Gamma Doppler (Factory New)": 12000000,
    "★ Shadow Daggers | Marble Fade (Factory New)": 5500000,
    "★ Driver Gloves | Crimson Weave (Field-Tested)": 14000000,
    "★ Hand Wraps | Slaughter (Field-Tested)": 9000000,
    "★ Falchion Knife | Doppler (Factory New)": 6500000
}
rarity_map = {
    "Consumer Grade": "#B0B0B0",
    "Industrial Grade": "#5DADE2",
    "Mil-Spec": "#2874A6",
    "Restricted": "#8E44AD",
    "Classified": "#E91E63",
    "Covert": "#E74C3C",
    "Contraband": "#F1C40F",
    "Knife/Gloves": "#FFD700"
}
def get_rarity_color(skin_name):
    rarity = fetch_rarity(skin_name)
    return rarity_map.get(rarity, "#888888")
def fetch_rarity(skin_name):
    # Ưu tiên nhận dạng qua ký hiệu Knife/Gloves
    if "★" in skin_name:
        return "Knife/Gloves"

    # Nhận dạng Covert theo danh sách gốc
    covert_keywords = ["Neo-Noir", "Printstream"]
    if any(k in skin_name for k in covert_keywords):
        return "Covert"

    # Nếu chưa phân tích được → Unknown
    return "Unknown"


mock_float = {
    "AWP | Neo-Noir (Minimal Wear)": 0.12,
    "M4A4 | Neo-Noir (Field-Tested)": 0.18,
    "Glock-18 | Neo-Noir (Field-Tested)": 0.25,
    "USP-S | Neo-Noir (Field-Tested)": 0.18,
    "Desert Eagle | Printstream (Field-Tested)": 0.20
}

mock_pattern = {
    "AWP | Neo-Noir (Minimal Wear)": "Pattern A",
    "M4A4 | Neo-Noir (Field-Tested)": "Pattern B",
    "Glock-18 | Neo-Noir (Field-Tested)": "Pattern C",
    "USP-S | Neo-Noir (Field-Tested)": "Pattern A",
    "Desert Eagle | Printstream (Field-Tested)": "Pattern B"
}

history_file = "lich_su_gia_skin_30ngay.csv"

# ================== HÀM HỖ TRỢ ==================
def create_api_url(skin_name):
    base_url = "https://steamcommunity.com/market/priceoverview/"
    params = {"appid": 730, "currency": 1, "market_hash_name": skin_name}
    return f"{base_url}?{urlencode(params)}"

def fetch_steam_price(market_hash_name):
    """
    Trả về (lowest, median) dưới dạng float (USD). Trả (0,0) khi lỗi.
    Nhận market_hash_name (ví dụ: "AWP | Neo-Noir (Minimal Wear)").
    """
    url = f"https://steamcommunity.com/market/priceoverview/?appid=730&currency=1&market_hash_name={market_hash_name}"
    try:
        res = requests.get(url, timeout=6)
        res.raise_for_status()
        data = res.json()
        if not data.get("success"):
            return 0.0, 0.0
        lowest = normalize_price(data.get("lowest_price"))
        median = normalize_price(data.get("median_price"))
        return (lowest or 0.0), (median or 0.0)
    except Exception as e:
        print(f"[fetch_steam_price] Lỗi fetch Steam cho `{market_hash_name}`: {e}")
        return 0.0, 0.0

def normalize_price(price_str):
    if not price_str: return None
    price = price_str.replace("€", "").replace("£", "").replace("¥", "").replace("₫", "")
    price = price.replace("$","").replace(",",".").strip()
    try:
        return float(price)
    except:
        return None

def convert_usd_to_vnd(usd_amount):
    """
    Nhận float (ví dụ 1.23) hoặc chuỗi '$1.23' và trả int VND.
    Trả 0 nếu input không hợp lệ.
    """
    rate = 24000
    if usd_amount is None:
        return 0
    # nếu đã là số
    if isinstance(usd_amount, (int, float, np.floating)):
        try:
            return int(float(usd_amount) * rate)
        except:
            return 0
    # nếu là chuỗi, loại bỏ ký tự
    try:
        s = str(usd_amount).replace("$","").replace(",","").strip()
        val = float(s)
        return int(val * rate)
    except:
        return 0


# ================== LOAD DỮ LIỆU ==================
def load_data():
    # nếu chưa có file lịch sử thì trả empty dataframe với cột chuẩn
    cols = [
        "Ngày", "Tên Skin", "Giá Hiện Tại (VND)", "Giá TB 7 Ngày (VND)", "Thay Đổi %",
        "Lợi Nhuận %", "Tín Hiệu", "Float", "Pattern", "Giá Mục Tiêu (VND)",
        "Max 30 Ngày", "Min 30 Ngày", "Cảnh Báo Sideway", "Gợi ý",
        "Rủi ro", "Thanh khoản"
    ]
    if not os.path.exists(history_file):
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(history_file, encoding='utf-8-sig')
    # chuyển ngày về datetime
    if "Ngày" in df.columns:
        df["Ngày"] = pd.to_datetime(df["Ngày"], errors="coerce")
    return df

# ================== UPDATE (lưu lịch sử, không duplicate) ==================
def update_market_data():
    """Lấy dữ liệu hiện tại cho tất cả skins, append vào lịch sử,
       rồi giữ tối đa 30 bản ghi gần nhất cho mỗi skin."""
    records = []
    df = load_data()
    for skin in skins:
        url = create_api_url(skin)
        current, median = fetch_steam_price(url)
        current_val = convert_usd_to_vnd(current)
        median_val = convert_usd_to_vnd(median)
        # Nếu giá == 0 → dùng giá cũ gần nhất trong lịch sử
        if current_val == 0 or median_val == 0:
            old = df[df["Tên Skin"] == skin]
            if not old.empty:
                current_val = int(old["Giá Hiện Tại (VND)"].iloc[-1])
                median_val = int(old["Giá TB 7 Ngày (VND)"].iloc[-1])
            else:
                # fallback minimum
                current_val = median_val = target_prices.get(skin, 1000000)

        # Tính lại delta sau khi sửa giá
        delta = round((current_val - median_val) / median_val * 100, 2) if median_val > 0 else 0

        target = target_prices.get(skin, 1000000)
        profit_pct = round((current_val - target) / target * 100, 2) if target>0 else 0
        suggestion = "MUA" if profit_pct > 10 else ("BÁN" if profit_pct < -5 else "Chờ")
    # Lấy độ hiếm skin
        rarity = fetch_rarity(skin)
        rarity_color = rarity_map.get(rarity, "#FFD700")

        # gather
        records.append({
            "Ngày": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Tên Skin": skin,
            "Giá Hiện Tại (VND)": int(current_val),
            "Giá TB 7 Ngày (VND)": int(median_val),
            "Thay Đổi %": delta,
            "Lợi Nhuận %": profit_pct,
            "Rarity": rarity,
            "Rarity Màu": rarity_color,
            "Tín Hiệu": "",
            "Float": mock_float.get(skin, 0.2),
            "Pattern": mock_pattern.get(skin, "Unknown"),
            "Giá Mục Tiêu (VND)": target,
            "Max 30 Ngày": current_val,
            "Min 30 Ngày": current_val,
            "Cảnh Báo Sideway": "",
            "Gợi ý": suggestion,
            "Rủi ro": 1/(mock_float.get(skin,0.2) or 0.2),
            "Thanh khoản": 50
        })
        time.sleep(0.3)
    new_df = pd.DataFrame(records)
    if os.path.exists(history_file):
        old = pd.read_csv(history_file, encoding='utf-8-sig')
        combined = pd.concat([old, new_df], ignore_index=True)
    else:
        combined = new_df
    # giữ 30 bản ghi cuối mỗi skin
    combined["Ngày"] = pd.to_datetime(combined["Ngày"], errors="coerce")
    combined = combined.sort_values(["Tên Skin", "Ngày"])
    combined = combined.groupby("Tên Skin").tail(30).reset_index(drop=True)
    combined.to_csv(history_file, index=False, encoding='utf-8-sig')
    return combined

# ================== AI PREDICTION 7 ngày (Linear Regression) ==================
def predict_7d(df_skin, date_col="Ngày", price_col="Giá Hiện Tại (VND)"):
    """
    Dự đoán 7 ngày tới bằng mô hình Prophet.
    Trả về dataframe chứa: ds, yhat, yhat_lower, yhat_upper.
    """
    df = df_skin.dropna(subset=[price_col]).copy()
    df = df.rename(columns={date_col: "ds", price_col: "y"})
    if len(df) < 5:
        return None  # không đủ dữ liệu

    try:
        model = Prophet(daily_seasonality=True, yearly_seasonality=False)
        model.fit(df)
        future = model.make_future_dataframe(periods=7)
        forecast = model.predict(future)
        return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(7)
    except Exception as e:
        st.warning(f"Lỗi dự đoán Prophet: {e}")
        return None


# ================== BACKTESTING đơn giản ==================
def backtest_simple(df_skin, take_profit=0.10, stop_loss=0.05, window_ma=7):
    """Chiến lược:
       - Buy khi giá hiện < rolling_mean(window)*0.98 (giả sử giá giảm về MA)
       - Sell khi đạt take_profit OR stop_loss
       Trả về dict chứa trades, metrics.
    """
    df = df_skin.sort_values("Ngày").copy()
    df["Close"] = df["Giá Hiện Tại (VND)"].astype(float)
    if df.empty:
        return None
    df["MA"] = df["Close"].rolling(window=window_ma, min_periods=1).mean()
    holding = False
    buy_price = 0
    trades = []
    for idx, row in df.iterrows():
        price = row["Close"]
        ma = row["MA"]
        date = row["Ngày"]
        if not holding:
            if price < ma * 0.98:
                # buy signal
                holding = True
                buy_price = price
                trades.append({"action":"BUY","date":date,"price":price})
        else:
            # check TP or SL
            if price >= buy_price * (1 + take_profit):
                holding = False
                trades.append({"action":"SELL","date":date,"price":price})
                buy_price = 0
            elif price <= buy_price * (1 - stop_loss):
                holding = False
                trades.append({"action":"SELL_STOP","date":date,"price":price})
                buy_price = 0
    # metrics
    wins = 0; losses = 0; gross_profit = 0
    # pair trades BUY->SELL
    i = 0
    while i < len(trades):
        if trades[i]["action"]=="BUY":
            j = i+1
            while j < len(trades) and trades[j]["action"].startswith("BUY"):
                j += 1
            if j < len(trades) and trades[j]["action"].startswith("SELL"):
                buy_p = trades[i]["price"]
                sell_p = trades[j]["price"]
                pnl = sell_p - buy_p
                gross_profit += pnl
                if pnl > 0: wins += 1
                else: losses += 1
                i = j+1
            else:
                i += 1
        else:
            i += 1
    total_trades = sum(1 for t in trades if t["action"].startswith("SELL"))
    metric = {
        "trades": trades,
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "gross_profit_vnd": int(gross_profit),
        "expected_return_per_trade": (gross_profit/total_trades) if total_trades>0 else 0
    }
    # suggestion simple: nếu expected_return_per_trade > 0 and wins/total_trades > 0.5 -> MUA
    suggestion = "Chờ"
    if metric["total_trades"]>0:
        win_rate = metric["wins"]/metric["total_trades"] if metric["total_trades"]>0 else 0
        if metric["expected_return_per_trade"] > 0 and win_rate >= 0.5:
            suggestion = "GỢI Ý MUA (backtest)"
    metric["suggestion"] = suggestion
    return metric

# ================== PLOT PREDICTION OVERLAY (Plotly) ==================
def plot_history_and_prediction(df_skin, preds_df, date_col="Ngày", price_col="Giá Hiện Tại (VND)", title="History + 7d Forecast"):
    df = df_skin.sort_values(date_col)
    color = get_rarity_color(df_skin["Tên Skin"].iloc[0]) if not df_skin.empty else "#999999"
    fig = go.Figure()

    # Thực tế
    fig.add_trace(go.Scatter(
        x=df[date_col], y=df[price_col], 
        mode='lines+markers', name='Thực tế', line=dict(color=color, width=2)
    ))

    # Dự báo (Prophet)
    if preds_df is not None:
        fig.add_trace(go.Scatter(
            x=preds_df["ds"], y=preds_df["yhat"], 
            mode='lines', name='Dự báo', line=dict(color='orange', dash='dash')
        ))
        fig.add_trace(go.Scatter(
            x=pd.concat([preds_df["ds"], preds_df["ds"][::-1]]),
            y=pd.concat([preds_df["yhat_upper"], preds_df["yhat_lower"][::-1]]),
            fill='toself', fillcolor='rgba(255,165,0,0.2)',
            line=dict(color='rgba(255,255,255,0)'), name='Khoảng dự báo'
        ))

    fig.update_layout(title=title, xaxis_title="Ngày", yaxis_title="Giá (VND)", height=550, template="plotly_white")
    return fig
def auto_col_widths(data, font_name='DejaVuSans', font_size=9):
    """
    Tính độ rộng tối thiểu của mỗi cột dựa theo nội dung dài nhất.
    """
    num_cols = len(data[0])
    widths = []
    for col in range(num_cols):
        max_width = 0
        for row in data:
            text = str(row[col])
            w = stringWidth(text, font_name, font_size)
            if w > max_width:
                max_width = w
        # Cộng thêm padding 1 chút
        widths.append(max_width + 10)
    return widths
# ================== PDF CREATE (sửa lỗi: trả về file path và đảm bảo tạo xong trước khi download) ==================
def create_pdf(df_input):
    """
    Xuất PDF Dashboard Skin Steam bằng ReportLab (hỗ trợ tiếng Việt hoàn chỉnh)
    - Fix: kiểm tra rỗng, tránh lỗi biến 'data'
    - Tự động giãn cột theo nội dung (auto-fit)
    - Font DejaVuSans Unicode đầy đủ (có hỗ trợ tiếng Việt)
    """
    # ====== CHUẨN BỊ DỮ LIỆU ======
    if df_input is None or df_input.empty:
        raise ValueError("Không có dữ liệu để tạo PDF.")

    df_input = df_input.copy()
    df_input["Ngày"] = pd.to_datetime(df_input["Ngày"], errors="coerce")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    pdf_output = os.path.abspath(f"Report_{timestamp}.pdf")

    # ====== TÍNH TOÁN TỔNG HỢP ======
    total_value = int(df_input["Giá Hiện Tại (VND)"].sum())
    total_profit_vnd = int(((df_input["Lợi Nhuận %"] / 100) * df_input["Giá Hiện Tại (VND)"]).sum())
    mua_count = int((df_input["Gợi ý"] == "MUA").sum())
    ban_count = int((df_input["Gợi ý"] == "BÁN").sum())
    cho_count = int((df_input["Gợi ý"] == "Chờ").sum())

    # ====== VẼ BIỂU ĐỒ (Matplotlib) ======
    fig, ax = plt.subplots(figsize=(7, 4))
    for skin in df_input["Tên Skin"].unique():
        skin_data = df_input[df_input["Tên Skin"] == skin].sort_values("Ngày")
        ax.plot(skin_data["Ngày"], skin_data["Giá Hiện Tại (VND)"], marker="o", label=skin)
    ax.set_title("Biểu đồ lịch sử giá Skin", fontname="DejaVu Sans")
    ax.set_xlabel("Ngày", fontname="DejaVu Sans")
    ax.set_ylabel("Giá (VND)", fontname="DejaVu Sans")
    ax.legend(fontsize=7)
    plt.tight_layout()

    chart_file = os.path.abspath("chart_temp.png")
    fig.savefig(chart_file, dpi=150)
    plt.close(fig)

    # ====== CẤU HÌNH FONT & STYLE ======
    doc = SimpleDocTemplate(
        pdf_output, pagesize=A4,
        rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30
    )

    styles = getSampleStyleSheet()
    normal = ParagraphStyle('Normal', fontName='DejaVuSans', fontSize=11)
    title = ParagraphStyle('Title', parent=normal, fontName='DejaVuSans-Bold', fontSize=18, alignment=1)
    body = ParagraphStyle('Body', parent=normal, fontSize=11, leading=14)
    story = []

    # ====== HEADER ======
    story.append(Paragraph("<b>Báo cáo Dashboard Skin Steam</b>", title))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(f"Thời gian xuất: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", body))
    story.append(Paragraph(f"Tổng giá trị: <b>{total_value:,} VND</b>", body))
    story.append(Paragraph(f"Tổng lợi nhuận ước tính: <b>{total_profit_vnd:,} VND</b>", body))
    story.append(Paragraph(f"Số lượng gợi ý: MUA {mua_count} — BÁN {ban_count} — CHỜ {cho_count}", body))
    story.append(Spacer(1, 0.5 * cm))

    # ====== BIỂU ĐỒ ======
    story.append(Image(chart_file, width=16 * cm, height=8 * cm))
    story.append(Spacer(1, 0.5 * cm))

    # ====== BẢNG DỮ LIỆU ======
    df_display = df_input[[
        "Tên Skin", "Giá Hiện Tại (VND)", "Giá TB 7 Ngày (VND)",
        "Lợi Nhuận %", "Gợi ý", "Float", "Pattern"
    ]].copy()
    data = [list(df_display.columns)] + df_display.values.tolist()
    if not data or len(data[0]) == 0:
        raise ValueError("Không có dữ liệu hợp lệ để tạo PDF.")

    col_widths = auto_col_widths(data, font_name='DejaVuSans', font_size=9)
    table = Table(data, colWidths=col_widths)
    table_style = TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "DejaVuSans"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#444444")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
    ])

    # Thêm màu xen kẽ dòng
    for i in range(1, len(data)):
        if i % 2 == 0:
            table_style.add("BACKGROUND", (0, i), (-1, i), colors.whitesmoke)
        else:
            table_style.add("BACKGROUND", (0, i), (-1, i), colors.beige)

    table.setStyle(table_style)
    story.append(table)

    # ====== FOOTER ======
    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph(
        "Báo cáo được tạo tự động bằng ứng dụng <b>Streamlit — CS2 Skin Tracker</b>",
        body
    ))

    # ====== XUẤT FILE ======
    doc.build(story)
    return pdf_output


# ================== STREAMLIT UI ==================

st.title("Theo dõi Skin Steam")

# Nút cập nhật (gọi update_market_data)
if st.button("Cập nhật dữ liệu"):
    try:
        update_market_data()
        st.success("Đã cập nhật dữ liệu lên lịch sử (file CSV).")
    except Exception as e:
        st.error(f"Cập nhật thất bại: {e}")

# Load data
df = load_data()

# Sidebar filter
# Sidebar filter
st.sidebar.subheader("Bộ lọc Skin")

available_skins = df["Tên Skin"].unique() if not df.empty else skins

# Vẫn dùng multiselect để lọc dữ liệu thật
skin_selected = st.sidebar.multiselect(
    "Chọn Skin để phân tích dữ liệu",
    options=available_skins,
    default=list(available_skins)
)

period_days = st.sidebar.selectbox("Chọn khoảng thời gian", [7, 14, 30], index=2)

# Bổ sung dropdown HTML chỉ để hiển thị màu theo độ hiếm
dropdown_html = """
<p><strong>Độ hiếm — hiển thị màu theo rarity:</strong></p>
<select multiple style='width: 100%; height: 220px; padding:6px;'>
"""

for s in available_skins:
    rarity = None
    if not df.empty and s in df["Tên Skin"].values:
        # lấy rarity mới nhất
        rarity = df[df["Tên Skin"] == s]["Rarity"].iloc[-1]
    color = rarity_map.get(rarity, "#888888")
    dropdown_html += (
        f"<option style='color:{color}; font-weight:bold;'>{s}</option>"
    )

dropdown_html += "</select>"

st.sidebar.markdown(dropdown_html, unsafe_allow_html=True)


# filtered df (keep last period_days per skin)
if df.empty:
    df_filtered = pd.DataFrame()
else:
    df_filtered = df[df["Tên Skin"].isin(skin_selected)].copy()
    df_filtered = df_filtered.sort_values(["Tên Skin", "Ngày"])
    df_filtered = df_filtered.groupby("Tên Skin").tail(period_days).reset_index(drop=True)

# enrich ROI (gross/net) for display
def compute_roi_row(price_vnd, target_vnd, fee=0.15):
    try:
        roi_gross = (price_vnd - target_vnd) / target_vnd * 100
    except:
        roi_gross = 0.0
    cost_with_fee = target_vnd * (1 + fee)
    proceeds_with_fee = price_vnd * (1 - fee)
    try:
        roi_net = (proceeds_with_fee - cost_with_fee) / cost_with_fee * 100
    except:
        roi_net = 0.0
    return round(roi_gross,2), round(roi_net,2), int(price_vnd - target_vnd), int(proceeds_with_fee - cost_with_fee)

if not df_filtered.empty:
    gg = df_filtered.apply(lambda r: compute_roi_row(r["Giá Hiện Tại (VND)"], r["Giá Mục Tiêu (VND)"]), axis=1)
    df_filtered["ROI Gross %"] = [x[0] for x in gg]
    df_filtered["ROI Net %"] = [x[1] for x in gg]
    df_filtered["Lợi nhuận (Gross VND)"] = [x[2] for x in gg]
    df_filtered["Lợi nhuận (Net VND)"] = [x[3] for x in gg]

# ================== TÍNH KPI ==================
if not df_filtered.empty:
    total_value = int(df_filtered["Giá Hiện Tại (VND)"].sum())
    total_profit_vnd = int(((df_filtered["Lợi Nhuận %"]/100) * df_filtered["Giá Hiện Tại (VND)"]).sum())
else:
    total_value = 0
    total_profit_vnd = 0

st.markdown(f"**Tổng giá trị:** {total_value:,.0f} VND — **Tổng lợi nhuận ước tính:** {total_profit_vnd:,.0f} VND")

# ================== Gợi ý tốt nhất (safely) ==================
if not df_filtered.empty:
    try:
        best_deal_idx = df_filtered["Lợi Nhuận %"].idxmax()
    except Exception:
        best_deal_idx = None
    if best_deal_idx is not None and best_deal_idx in df_filtered.index:
        df_filtered["Highlight"] = df_filtered.index.map(lambda x: "Gợi ý tốt" if x==best_deal_idx else "")
    else:
        df_filtered["Highlight"] = ""

# display table
def highlight_profit(val):
    color = "green" if val>10 else "red" if val<0 else ""
    return f"background-color: {color}"

def highlight_sideway(val):
    color = "yellow" if val=="Sideway – Cảnh Báo" else ""
    return f"background-color: {color}"
def apply_rarity_color(row):
    rarity = row.get("Rarity", None)
    color = rarity_map.get(rarity, "#888888")
    return [f"color:{color}; font-weight:bold;" if col == "Tên Skin" else "" for col in row.index]

st.subheader("Dashboard Skin Steam Nâng Cao")
if df_filtered.empty:
    st.info("Chưa có dữ liệu. Nhấn 'Cập nhật dữ liệu' để thu thập.")
else:
    st.dataframe(
    df_filtered.style
    .apply(apply_rarity_color, axis=1)
    .applymap(highlight_profit, subset=["Lợi Nhuận %"])
    .applymap(highlight_sideway, subset=["Cảnh Báo Sideway"])
)


# ================== Candlestick + Prediction + Backtest per skin ==================
st.subheader("Phân tích nâng cao — Dự đoán 7 ngày & Backtest")
skin_to_show = st.selectbox("Chọn skin để phân tích", df_filtered["Tên Skin"].unique() if not df_filtered.empty else [])
if skin_to_show:
    hist = pd.read_csv(history_file, encoding='utf-8-sig')
    hist["Ngày"] = pd.to_datetime(hist["Ngày"], errors="coerce")
    skin_hist = hist[hist["Tên Skin"]==skin_to_show].sort_values("Ngày").tail(90)
    if skin_hist.empty:
        st.info("Chưa có lịch sử cho skin này. Nhấn 'Cập nhật dữ liệu' để lấy mới.")
    else:
        preds = predict_7d(skin_hist, date_col="Ngày", price_col="Giá Hiện Tại (VND)")
        fig_pred = plot_history_and_prediction(
            skin_hist, preds, 
            date_col="Ngày", price_col="Giá Hiện Tại (VND)", 
            title=f"{skin_to_show} — Dự báo giá 7 ngày (Prophet)"
        )
        st.plotly_chart(fig_pred, use_container_width=True)

        # Backtest
        bt = backtest_simple(skin_hist)
        if bt is None:
            st.info("Không đủ dữ liệu để backtest.")
        else:
            st.markdown("**Kết quả backtest đơn giản:**")
            st.write(f"- Tổng giao dịch (sell): {bt['total_trades']}")
            st.write(f"- Wins: {bt['wins']}, Losses: {bt['losses']}")
            st.write(f"- Lợi nhuận gộp (VND): {bt['gross_profit_vnd']}")
            st.write(f"- Kỳ vọng/trade: {bt['expected_return_per_trade']:.2f} VND")
            st.write(f"- Gợi ý (theo backtest): {bt.get('suggestion','Chờ')}")
            # show trade list
            if bt["trades"]:
                st.table(pd.DataFrame(bt["trades"]))

# ================== BIỂU ĐỒ SO SÁNH NHIỀU SKIN ==================
st.subheader("So sánh giá nhiều skin")

skin_selected_multi = st.multiselect(
    "Chọn skin để so sánh",
    options=available_skins,
    default=list(available_skins)
)

if os.path.exists(history_file) and len(skin_selected_multi) > 0:
    hist = pd.read_csv(history_file, encoding="utf-8-sig")
    hist["Ngày"] = pd.to_datetime(hist["Ngày"], errors="coerce")

    fig = go.Figure()

    for skin in skin_selected_multi:
        sdata = hist[hist["Tên Skin"] == skin].sort_values("Ngày").tail(period_days)

        rarity = sdata["Rarity"].iloc[-1] if "Rarity" in sdata.columns and not sdata.empty else None
        color = rarity_map.get(rarity, "#999999")

        fig.add_trace(go.Scatter(
            x=sdata["Ngày"],
            y=sdata["Giá Hiện Tại (VND)"],
            mode='lines+markers',
            name=skin,
            line=dict(color=color, width=2)
        ))

    fig.update_layout(
        title=f"So sánh giá skin ({period_days} ngày gần nhất)",
        xaxis_title="Ngày",
        yaxis_title="Giá (VND)",
        template="plotly_white",
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)


# ================== RỦI RO VS THANH KHOẢN ==================
st.subheader("Rủi ro vs Thanh khoản")
if not df_filtered.empty:
    fig, ax = plt.subplots(figsize=(8,5))
    ax.scatter(df_filtered["Rủi ro"], df_filtered["Thanh khoản"], c=df_filtered["Lợi Nhuận %"], cmap="RdYlGn", s=100)
    for i, txt in enumerate(df_filtered["Tên Skin"]):
        ax.annotate(txt, (df_filtered["Rủi ro"].iloc[i], df_filtered["Thanh khoản"].iloc[i]))
    ax.set_xlabel("Rủi ro (Float thấp → ít rủi ro)")
    ax.set_ylabel("Thanh khoản (Lượt giao dịch)")
    ax.set_title("Rủi ro vs Thanh khoản skin")
    st.pyplot(fig)
else:
    st.info("Không có dữ liệu để vẽ Rủi ro vs Thanh khoản.")

# ================== EXPORT PDF (sửa lỗi download) ==================
if st.button("Xuất PDF Dashboard nâng cao"):
    pdf_file = None
    data = None
    filename = None

    if df_filtered.empty:
        st.warning("Không có dữ liệu để xuất PDF.")
    else:
        try:
            pdf_file = create_pdf(df_filtered)
            with open(pdf_file, "rb") as f:
                data = f.read()
            filename = os.path.basename(pdf_file)
            st.success(f"Đã tạo PDF: {filename}")
        except Exception as e:
            st.error(f"Lỗi khi xuất PDF: {e}")

    # 👉 Đặt ngoài khối try/except
    if data and filename:
        st.download_button(
            label="⬇ Tải PDF Dashboard",
            data=data,
            file_name=filename,
            mime="application/pdf"
        )


