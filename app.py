import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
from datetime import datetime
from dotenv import load_dotenv

from data_manager import sync_csv_to_firestore, get_user_transactions
from rag_engine import RAGEngine

# Load environment variables
load_dotenv()

# --- Page Config ---
st.set_page_config(
    page_title="Finance Tracker AI",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- STATE MANAGEMENT ---
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

if 'current_view' not in st.session_state:
    st.session_state.current_view = 'Home'
    
if 'category_filter' not in st.session_state:
    st.session_state.category_filter = None

if 'data' not in st.session_state:
    # Initial Mock Data Load (will be replaced by CSV if uploaded)
    dates = pd.date_range(end=datetime.today(), periods=30)
    data = {
        "Date": dates,
        "Description": ["Salary", "Rent", "Groceries", "Uber", "Netflix", "Dividends", "Gym", "Dining Out", "Flight", "Consulting"] * 3,
        "Amount": [5000, 2000, 300, 45, 15, 150, 60, 120, 400, 1500] * 3,
        "Category": ["Income", "Rental", "Grocery", "Travel", "Entertainment", "Investments", "Health", "Food", "Travel", "Business"] * 3,
        "Type": ["Income", "Expense", "Expense", "Expense", "Expense", "Income", "Expense", "Expense", "Expense", "Income"] * 3
    }
    st.session_state.data = pd.DataFrame(data)
    st.session_state.is_demo = True

    if 'rag' not in st.session_state:
        st.session_state.rag = RAGEngine()
        st.session_state.rag.ingest_data(st.session_state.data)

df = st.session_state.data

# --- CONSTANTS & HELPERS ---
CURRENCY_SYMBOL = "₹"
CURRENT_DATE = datetime.now()
CURRENT_MONTH_NAME = CURRENT_DATE.strftime("%B")

def format_inr(amount):
    return f"{CURRENCY_SYMBOL}{amount:,.0f}"

# --- CSS STYLING (Premium Glassmorphism + Better Dark Mode) ---
is_dark = st.session_state.theme == 'dark'

# Enhanced Palette
bg_color = "#0E1117" if is_dark else "#F8F9FA" # Midnight Blue for Dark Mode
card_bg = "#1E202B" if is_dark else "rgba(255, 255, 255, 0.95)" # Dark Blue-Grey
text_main = "#FAFAFA" if is_dark else "#1A1A1A"
text_sub = "#B0B0B0" if is_dark else "#636E72"
shadow = "0 8px 24px rgba(0,0,0,0.5)" if is_dark else "0 4px 20px rgba(0,0,0,0.08)"
border = "1px solid #333" if is_dark else "1px solid #E5E5E5"

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Outfit', sans-serif;
        background-color: {bg_color};
        color: {text_main};
    }}
    
    .stApp {{
        background-color: {bg_color};
    }}
    
    /* GLASSMORPHISM CARD */
    .premium-card {{
        background: {card_bg};
        border-radius: 20px;
        padding: 24px;
        box-shadow: {shadow};
        border: {border};
        backdrop-filter: blur(10px);
        margin-bottom: 20px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }}
    .premium-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.12);
    }}

    /* HEADER GRADIENT */
    .header-mesh {{
        background: linear-gradient(120deg, #E3F2FD 0%, #F3E5F5 100%);
        padding: 20px;
        border-radius: 0 0 30px 30px;
        margin-bottom: 30px;
        -webkit-mask-image: radial-gradient(white, black);
    }}
    
    /* TYPOGRAPHY */
    .big-balance {{
        font-size: 3.5rem;
        font-weight: 800;
        color: {text_main};
        line-height: 1.1;
    }}
    .label-text {{
        font-weight: 600;
        color: {text_sub};
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}

    /* CHAT BUBBLES */
    .chat-user {{
        background: #F1F3F4;
        color: #2D3436;
        padding: 12px 18px;
        border-radius: 18px 18px 0 18px;
        margin: 5px 0 5px auto;
        max-width: 80%;
    }}
    .chat-ai {{
        background: linear-gradient(135deg, #E0F7FA 0%, #E1BEE7 100%);
        color: #2D3436;
        padding: 12px 18px;
        border-radius: 18px 18px 18px 0;
        margin: 5px 0;
        max-width: 80%;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }}
    
    /* Highlights */
    .highlight {{
         color: { "#FFD700" if is_dark else "#FBC02D" };
    }}

    /* ANIM */
    @keyframes float {{
        0% {{ transform: translateY(0px); }}
        50% {{ transform: translateY(-5px); }}
        100% {{ transform: translateY(0px); }}
    }}
    .float-icon {{
        animation: float 3s ease-in-out infinite;
    }}

    /* Hide Elements */
    header, footer {{visibility: hidden;}}
    [data-testid="stSidebar"] {{ display: none; }}
</style>
""", unsafe_allow_html=True)


# --- VIEW: HOME ---
def render_home():
    # 1. Custom Header
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown(f"""
        <div style="padding: 10px 0;">
            <h1 style="margin:0; font-size: 2.2rem; background: -webkit-linear-gradient(left, #1E88E5, #9C27B0); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Good Morning, {st.session_state.get('user_name', 'Ajith')}!</h1>
            <p style="margin:0; color:#95A5A6; font-size:1rem;">{CURRENT_DATE.strftime('%A, %d %B %Y')}</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
         if st.button("🔄 Refresh"):
             st.rerun()

    # 2. Dynamic Balance Card
    total_income = df[df['Type'] == 'Income']['Amount'].sum()
    total_expense = df[df['Type'] == 'Expense']['Amount'].sum()
    total_balance = total_income - total_expense
    
    st.markdown(f"""
    <div class="premium-card" style="position:relative;">
        <div style="display:flex; justify-content:space-between; align-items:start;">
            <div>
                <div class="label-text">Total Balance</div>
                <div class="big-balance">{format_inr(total_balance)}</div>
            </div>
            <div style="text-align:right;">
                <div style="background:#E8F5E9; color:#2E7D32; padding:5px 12px; border-radius:20px; font-weight:700;">
                     Create Wealth 🚀
                </div>
            </div>
        </div>
        <div style="margin-top:20px; display:flex; gap:10px;">
            <span style="background:#E8F5E9; color:#2E7D32; padding:5px 15px; border-radius:15px; font-weight:600; border:1px solid #C8E6C9;">Income: {format_inr(total_income)}</span>
            <span style="background:#FFEBEE; color:#C62828; padding:5px 15px; border-radius:15px; font-weight:600; border:1px solid #FFCDD2;">Spend: {format_inr(total_expense)}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 3. Financial Trends (Dynamic Chart)
    st.markdown("### Financial Trends")
    
    df['Date'] = pd.to_datetime(df['Date'])
    # Net Savings per Month
    df['Month_Num'] = df['Date'].dt.to_period('M')
    chronological_months = sorted(df['Month_Num'].unique().astype(str))
    
    months_labels = []
    savings_values = []
    
    for m in chronological_months:
        mask = df['Date'].dt.to_period('M').astype(str) == m
        m_df = df[mask]
        inc = m_df[m_df['Type'] == 'Income']['Amount'].sum()
        exp = m_df[m_df['Type'] == 'Expense']['Amount'].sum()
        
        label = pd.to_datetime(m).strftime('%b')
        months_labels.append(label)
        savings_values.append(inc - exp)

    fig = go.Figure(data=[go.Bar(
        x=months_labels,
        y=savings_values,
        text=[format_inr(x) for x in savings_values],
        textposition='outside', 
        marker=dict(
            color=savings_values,
            colorscale='Tealgrn', 
            showscale=False
        )
    )])
    fig.update_layout(
        font=dict(family="Outfit", color=text_sub),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=10, b=0),
        yaxis=dict(title="Net Savings (₹)", showgrid=True, gridcolor='rgba(0,0,0,0.05)'),
        xaxis=dict(showgrid=False),
        height=300,
        dragmode=False
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    # 4. Insight & Spending Cards (Top Categories)
    expense_df = df[df['Type'] == 'Expense']
    if not expense_df.empty:
        top_cat = expense_df.groupby('Category')['Amount'].sum().idxmax()
        top_cat_amt = expense_df.groupby('Category')['Amount'].sum().max()
    else:
        top_cat = "None"
        top_cat_amt = 0

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="premium-card" style="background: linear-gradient(135deg, #F0F4C3 0%, #FFFFFF 100%);">
            <div style="position:absolute; top:15px; right:15px; font-size:1.5rem;" class="float-icon">💡</div>
            <div class="label-text" style="color:#827717;">Top Expense</div>
            <div style="font-size:1.5rem; font-weight:700; color:#333; margin-top:10px;">
                {top_cat}
            </div>
            <div style="font-size:0.9rem; color:#666; margin-top:5px;">Total: {format_inr(top_cat_amt)}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="premium-card" style="background: linear-gradient(135deg, #FFEBEE 0%, #FFFFFF 100%);">
            <div style="position:absolute; top:15px; right:15px; font-size:1.5rem;" class="float-icon">📉</div>
            <div class="label-text" style="color:#C62828;">Spending Status</div>
            <div style="font-size:1.5rem; font-weight:700; color:#333; margin-top:10px;">
                {len(df)} Txns
            </div>
            <div style="font-size:0.9rem; color:#666; margin-top:5px;">Tracked this period</div>
        </div>
        """, unsafe_allow_html=True)

    # 5. Recent Activity (Dynamic & Safe)
    st.markdown("### Recent Activity")
    
    col_act, col_btn = st.columns([3, 1])
    with col_act:
        st.write("Latest transactions overview")
    with col_btn:
        if st.button("Check All ➔"):
            st.session_state.current_view = 'Analytics'
            st.rerun()

    recent_df = df.sort_values('Date', ascending=False).head(3)
    
    for idx, row in recent_df.iterrows():
        # SAFE STRING CASTING TO FIX CRASH
        cat_str = str(row['Category']).lower()
        desc_str = str(row['Description']).lower()
        type_str = str(row['Type']).lower()
        
        icon = "💸"
        if "food" in cat_str: icon = "🍔"
        elif "travel" in cat_str: icon = "✈️"
        elif "income" in type_str: icon = "💰"
        
        color_class = "#E53935" if row['Type'] == 'Expense' else "#2E7D32"
        sign = "-" if row['Type'] == 'Expense' else "+"
        
        st.markdown(f"""
        <div class="premium-card" style="padding:15px; display:flex; align-items:center; margin-bottom:10px;">
             <div style="background:#F3E5F5; color:#8E24AA; width:45px; height:45px; border-radius:50%; display:flex; align-items:center; justify-content:center; margin-right:15px; font-size:1.2rem;">
                {icon}
             </div>
             <div style="flex-grow:1;">
                 <div style="font-weight:700; color:{text_main};">{row['Description']}</div>
                 <div style="font-size:0.8rem; color:{text_sub};">{row['Date'].strftime('%d %b')} • {row['Category']}</div>
             </div>
             <div style="font-weight:700; color:{color_class};">{sign} {format_inr(row['Amount'])}</div>
        </div>
        """, unsafe_allow_html=True)

# --- VIEW: ANALYTICS ---
def render_analytics():
    st.markdown("## Analytics Dashboard")
    
    # 1. Real Date Picker
    c1, c2 = st.columns([1, 3])
    with c1:
        # Default to latest month in data
        default_date = df['Date'].max() if not df.empty else datetime.today()
        selected_month = st.date_input("Filter Month", value=default_date)
        
        # Filter by Month & Year
        mask = (df['Date'].dt.year == selected_month.year) & (df['Date'].dt.month == selected_month.month)
        month_df = df[mask]
    
    # 2. Donut (Pie) Chart
    st.markdown("### Spending Distribution")
    
    if month_df.empty:
        st.warning(f"No transactions found for {selected_month.strftime('%B %Y')}.")
        cat_data = pd.DataFrame() # Empty
    else:
        cat_data = month_df.groupby('Category')['Amount'].sum().reset_index()
        
        # Donut Chart for better aesthetics
        fig = px.pie(
            cat_data, 
            values='Amount', 
            names='Category',
            hole=0.5, # Donut style
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        
        # Update styling for dark/light
        fig.update_traces(textinfo='percent+label', textfont_size=13, hoverinfo="label+percent+name+value")
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Outfit", color=text_main, size=14),
            showlegend=True,
            legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="right", x=1.1 if not is_dark else 1.2),
            margin=dict(t=20, b=20, l=0, r=0),
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    # 3. Filter Buttons (Categories) (Dynamic)
    st.write("---")
    st.markdown("### Detailed Breakdown")
    
    if not df.empty:
        top_cats = df['Category'].value_counts().head(3).index.tolist()
    else:
        top_cats = []

    cols = st.columns(3)
    
    for idx, cat in enumerate(top_cats):
        cat_sum = df[df['Category'] == cat]['Amount'].sum()
        with cols[idx]:
            colors = ["#E3F2FD", "#F3E5F5", "#E8F5E9"]
            c_code = colors[idx % 3] if not is_dark else "#262730"
            border_c = "none" if not is_dark else "1px solid #444"
            
            st.markdown(f"""
            <div style="background:{c_code}; padding:15px; border-radius:15px; margin-bottom:10px; text-align:center; border:{border_c};">
                <div style="color:{text_sub}; font-size:0.9rem;">{cat}</div>
                <div style="color:{text_main}; font-weight:800; font-size:1.2rem;">{format_inr(cat_sum)}</div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button(f"Filter {cat}", key=f"btn_filter_{cat}"):
                st.session_state.category_filter = cat
                st.rerun()

    # Filtered Table
    if st.session_state.category_filter:
        st.markdown(f"#### Viewing: {st.session_state.category_filter}")
        f_df = df[df['Category'] == st.session_state.category_filter]
        
        if f_df.empty:
             st.warning(f"No transactions found for {st.session_state.category_filter}.")
        else:
             st.dataframe(f_df, use_container_width=True)
             
        if st.button("Clear Filter"):
            st.session_state.category_filter = None
            st.rerun()

# --- VIEW: AI ADVISOR ---
def render_ai_advisor():
    st.markdown("## 🤖 AI Financial Advisor")
    
    chat_container = st.container(height=500)
    prompt = st.chat_input("Ask: 'How much did I spend this month?'")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    with chat_container:
        for msg in st.session_state.messages:
            div_class = "chat-user" if msg["role"] == "user" else "chat-ai"
            st.markdown(f'<div class="{div_class}">{msg["content"]}</div>', unsafe_allow_html=True)

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.rerun()

    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        with st.spinner("Analyzing data..."):
            system_context = f"Today's Date is {CURRENT_DATE}. Assume queries about 'this month' refer to {CURRENT_MONTH_NAME}. Always format currency in Indian Rupees ({CURRENCY_SYMBOL})."
            final_query = f"{system_context}\n\nUser Query: {st.session_state.messages[-1]['content']}"
            
            response = st.session_state.rag.query_llm(final_query)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

# --- VIEW: SETTINGS ---
def render_settings():
    st.markdown("## ⚙️ Settings")
    
    # Data Source Indicator
    is_demo = st.session_state.get('is_demo', True)
    source_color = "#FFA726" if is_demo else "#66BB6A"
    source_text = "Demo Data" if is_demo else "Uploaded Data"
    
    st.markdown(f"""
    <div class="premium-card" style="display:flex; align-items:center; justify-content:space-between; background: linear-gradient(to right, {card_bg}, {card_bg}); border-left: 5px solid {source_color};">
        <div>
            <div class="label-text">Current Data Source</div>
            <div style="font-size:1.2rem; font-weight:700;">{source_text}</div>
        </div>
        <div style="font-size:2rem;">{'🧪' if is_demo else '📂'}</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Data Import")
    uploaded_file = st.file_uploader("Upload CSV Transaction File", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df_new = pd.read_csv(uploaded_file)
            
            # Normalize headers
            df_new.columns = [c.strip().title() for c in df_new.columns]
            
            # Revised Validation
            required_cols = {'Date', 'Description', 'Amount', 'Category'}
            
            if required_cols.issubset(df_new.columns):
                if st.button("Load and Sync Data"):
                    # 1. Parse Date
                    df_new['Date'] = pd.to_datetime(df_new['Date'])
                    
                    # 2. Auto-generate 'Type' if missing
                    if 'Type' not in df_new.columns:
                        def infer_type(row):
                            cat = str(row['Category']).lower()
                            desc = str(row['Description']).lower()
                            if any(x in cat or x in desc for x in ['income', 'salary', 'dividend', 'profit']):
                                return 'Income'
                            return 'Expense'
                        df_new['Type'] = df_new.apply(infer_type, axis=1)
                    
                    # 3. Ensure 'Amount' is numeric
                    df_new['Amount'] = pd.to_numeric(df_new['Amount'], errors='coerce').fillna(0.0)

                    # Update State
                    st.session_state.data = df_new
                    st.session_state.is_demo = False
                    
                    # Re-ingest into RAG
                    with st.spinner("Updating AI Wisdom..."):
                         if 'rag' not in st.session_state:
                             st.session_state.rag = RAGEngine()
                         st.session_state.rag.ingest_data(df_new)
                    
                    st.success(f"Successfully loaded {len(df_new)} transactions!")
                    time.sleep(1)
                    st.rerun()
            else:
                missing = required_cols - set(df_new.columns)
                st.error(f"Missing columns: {', '.join(missing)}")
        except Exception as e:
            st.error(f"Error reading file: {e}")
            
    st.markdown("### Appearance")
    is_on = st.toggle("Dark Mode", value=(st.session_state.theme == 'dark'))
    if is_on and st.session_state.theme != 'dark':
        st.session_state.theme = 'dark'
        st.rerun()
    elif not is_on and st.session_state.theme != 'light':
        st.session_state.theme = 'light'
        st.rerun()

    st.markdown("### Profile")
    st.text_input("Display Name", value="Ajith")
    
    st.markdown("### Reset")
    if st.button("Reset to Demo Data"):
        st.cache_data.clear()
        st.session_state.data = get_premium_demo_data()
        st.session_state.is_demo = True
        st.session_state.rag.ingest_data(st.session_state.data)
        st.success("Reset to Demo Data!")
        time.sleep(1)
        st.rerun()

# --- MAIN NAVIGATION & RENDER ---
cols = st.columns(4)
with st.container():
    c1, c2, c3, c4 = st.columns(4)
    if c1.button("🏠 Home", use_container_width=True): st.session_state.current_view = 'Home'; st.rerun()
    if c2.button("📊 Analytics", use_container_width=True): st.session_state.current_view = 'Analytics'; st.rerun()
    if c3.button("🤖 AI", use_container_width=True): st.session_state.current_view = 'AI Advisor'; st.rerun()
    if c4.button("⚙️ Settings", use_container_width=True): st.session_state.current_view = 'Settings'; st.rerun()

st.divider()

if st.session_state.current_view == 'Home':
    render_home()
elif st.session_state.current_view == 'Analytics':
    render_analytics()
elif st.session_state.current_view == 'AI Advisor':
    render_ai_advisor()
elif st.session_state.current_view == 'Settings':
    render_settings()
