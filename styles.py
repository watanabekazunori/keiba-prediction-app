"""カスタムCSS定義 — ダークモード・ネオン・カード型 UI。"""

CUSTOM_CSS = """
<style>
/* 全体: ダークモード + 微妙なグラデ */
.stApp {
    background: linear-gradient(135deg, #0E1117 0%, #11151D 50%, #0F1419 100%);
}

/* ヘッダー: グラデ + ネオン */
.hero {
    background: linear-gradient(120deg, #FF6B35 0%, #FFA552 50%, #FFD700 100%);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    font-size: 3em;
    font-weight: 800;
    letter-spacing: -0.02em;
    line-height: 1.1;
    padding: 0;
    margin: 0;
}

.hero-sub {
    color: #B0B7C3;
    font-size: 1.05em;
    margin-top: 8px;
    margin-bottom: 24px;
    line-height: 1.5;
}

/* メトリクスカード */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #1A1F2E 0%, #20263A 100%);
    border: 1px solid rgba(255, 107, 53, 0.15);
    border-radius: 12px;
    padding: 16px 18px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.3);
    transition: transform 0.2s, box-shadow 0.2s;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(255, 107, 53, 0.2);
}
[data-testid="stMetricValue"] {
    color: #FFA552 !important;
    font-size: 1.6em !important;
    font-weight: 700;
}
[data-testid="stMetricLabel"] {
    color: #B0B7C3 !important;
    font-size: 0.85em !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* セクションタイトル */
h2 {
    color: #FAFAFA;
    border-left: 4px solid #FF6B35;
    padding-left: 12px;
    margin-top: 28px !important;
    font-weight: 700;
}
h3 {
    color: #E8ECEF;
    margin-top: 22px !important;
    font-weight: 600;
}

/* DataFrame */
.stDataFrame {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid rgba(255, 107, 53, 0.15);
    box-shadow: 0 4px 16px rgba(0,0,0,0.3);
}

/* ボタン */
.stButton > button {
    background: linear-gradient(135deg, #FF6B35 0%, #FFA552 100%);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 10px 24px;
    font-weight: 700;
    transition: all 0.2s;
    box-shadow: 0 4px 12px rgba(255, 107, 53, 0.3);
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 16px rgba(255, 107, 53, 0.5);
}

/* サイドバー */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0A0D14 0%, #11151D 100%);
    border-right: 1px solid rgba(255, 107, 53, 0.1);
}

/* セレクトボックス */
.stSelectbox > div > div {
    background-color: #1A1F2E !important;
    border-color: rgba(255, 107, 53, 0.25) !important;
    border-radius: 10px;
}

/* チップ・バッジ */
.badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 0.85em;
    font-weight: 600;
    margin-right: 6px;
    margin-bottom: 6px;
}
.badge-super { background: linear-gradient(135deg, #FF0844 0%, #FFB199 100%); color: white; }
.badge-high  { background: linear-gradient(135deg, #FF6B35 0%, #FFA552 100%); color: white; }
.badge-mid   { background: linear-gradient(135deg, #4A90E2 0%, #7FB3E8 100%); color: white; }
.badge-low   { background: linear-gradient(135deg, #5C6370 0%, #7C8290 100%); color: white; }
.badge-cold  { background: linear-gradient(135deg, #2C3142 0%, #4A4F5E 100%); color: #B0B7C3; }

/* レースカード */
.race-card {
    background: linear-gradient(135deg, #1A1F2E 0%, #20263A 100%);
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 16px;
    border: 1px solid rgba(255, 107, 53, 0.1);
    box-shadow: 0 4px 20px rgba(0,0,0,0.2);
}

/* expander */
[data-testid="stExpander"] {
    background: #11151D;
    border: 1px solid rgba(255, 107, 53, 0.15);
    border-radius: 10px;
    margin-bottom: 8px;
}

/* divider */
hr {
    border-color: rgba(255, 107, 53, 0.15);
    margin-top: 24px !important;
    margin-bottom: 24px !important;
}

/* スピナー */
[data-testid="stSpinner"] {
    color: #FF6B35;
}

/* タブ */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: #1A1F2E;
    border-radius: 8px 8px 0 0;
    padding: 8px 18px;
    color: #B0B7C3;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #FF6B35 0%, #FFA552 100%);
    color: white !important;
}

/* Plotly 透明背景 */
.js-plotly-plot .plotly {
    background: transparent !important;
}

/* スクロールバー */
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: #11151D; }
::-webkit-scrollbar-thumb { background: rgba(255, 107, 53, 0.4); border-radius: 5px; }
::-webkit-scrollbar-thumb:hover { background: rgba(255, 107, 53, 0.7); }
</style>
"""
