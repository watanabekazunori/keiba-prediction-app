"""keiba_app/app.py — 競馬予想 Web アプリ v2 (UI 刷新版)。

機能:
  - 🆕 これからのレース (自動取得済): 翌日全レース予想を場別フィルタ + 自信度ランキング
  - 📅 過去レース日付一覧: 自信度順
  - 🎯 race_id 個別予想: 全頭ランキング + 軸4点 + 詳細特徴量

UI:
  - ダークモード + ネオングラデ
  - Plotly ゲージチャート (自信度可視化)
  - Plotly 棒グラフ (全頭P_win比較)
  - カード型レイアウト
  - レスポンシブ
"""
from __future__ import annotations
import pickle
import json
import pathlib
import sys

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from styles import CUSTOM_CSS

APP_DIR = pathlib.Path(__file__).resolve().parent


# ───────────────────────────────────────────────────────────
# キャッシュ & ロード
# ───────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model():
    with open(APP_DIR / "model" / "keiba_v1.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_resource(show_spinner=False)
def load_meta():
    with open(APP_DIR / "model" / "keiba_v1_meta.json", "r") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_features():
    df = pd.read_parquet(APP_DIR / "data" / "features_v2.parquet")
    df["race_id"] = df["race_id"].astype(str)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_runs():
    jra = pd.read_parquet(APP_DIR / "data" / "runs_jra.parquet")
    nar = pd.read_parquet(APP_DIR / "data" / "runs_nar.parquet")
    df = pd.concat([jra, nar], ignore_index=True)
    df["race_id"] = df["race_id"].astype(str)
    df["umaban"] = pd.to_numeric(df["umaban"], errors="coerce").astype("Int64")
    return df


# ───────────────────────────────────────────────────────────
# 予想ロジック
# ───────────────────────────────────────────────────────────

def softmax_within(arr, temperature=1.0):
    arr = arr * temperature
    e = np.exp(arr - arr.max())
    return e / e.sum()


def plackett_luce_top_k(p_win, k=3):
    K = len(p_win); p = list(p_win); p_topk = [0.0] * K
    for i in range(K): p_topk[i] += p[i]
    if k <= 1: return p_topk
    for i in range(K):
        for j in range(K):
            if j == i: continue
            d = 1.0 - p[j]
            if d <= 1e-9: continue
            p_topk[i] += p[j] * (p[i] / d)
    if k <= 2: return p_topk
    for i in range(K):
        for j in range(K):
            if j == i: continue
            for m in range(K):
                if m == i or m == j: continue
                d1 = 1.0 - p[j]; d2 = 1.0 - p[j] - p[m]
                if d1 <= 1e-9 or d2 <= 1e-9: continue
                p_topk[i] += p[j] * (p[m] / d1) * (p[i] / d2)
    return [min(1.0, x) for x in p_topk]


def quinella_prob(pa, pb):
    da = 1.0 - pa; db = 1.0 - pb
    if da <= 1e-9 or db <= 1e-9: return 0.0
    return min(0.99, pa * (pb / da) + pb * (pa / db))


def estimate_show_payout(win_odds):
    if win_odds is None or pd.isna(win_odds) or win_odds <= 1.0: return 0.0
    return max(100.0, win_odds * 0.25 * 100.0)


def estimate_quinella_payout(oa, ob):
    if any(o is None or pd.isna(o) or o <= 1.0 for o in (oa, ob)): return 0.0
    return oa * ob * 0.5 * 100.0


def predict_race(race_id, bundle, feats, runs):
    race_id = str(race_id)
    df = feats[feats["race_id"] == race_id].copy()
    if df.empty:
        return {"error": f"race_id={race_id} のデータがありません。"}
    NUMERIC_FEATS = bundle["numeric_feats"]
    CATEGORICAL_FEATS = bundle["categorical_feats"]
    for c in CATEGORICAL_FEATS:
        if c in df.columns:
            df[c] = df[c].astype("category")
    X = df[NUMERIC_FEATS + CATEGORICAL_FEATS]
    p_lgb = bundle["lgb_clf"].predict_proba(X)[:, 1]
    X_cb = X.copy()
    for c in CATEGORICAL_FEATS:
        X_cb[c] = X_cb[c].astype(str).fillna("missing").replace({"nan":"missing","None":"missing"})
    p_cat = bundle["cb_clf"].predict_proba(X_cb)[:, 1]
    rk_score = bundle["rk"].predict(X)
    df["rk_score"] = rk_score
    df["p_rk"] = df.groupby("race_id")["rk_score"].transform(lambda x: softmax_within(x.to_numpy(), bundle["rk_softmax_temp"]))
    df["p_lgb"] = p_lgb; df["p_cat"] = p_cat
    w = bundle["blend_weights"]
    df["p_blend"] = w["lgb"]*df["p_lgb"] + w["cat"]*df["p_cat"] + w["rk"]*(df["p_rk"]*3.0).clip(0,1)
    df["cal_p"] = bundle["iso"].transform(df["p_blend"])
    df["P_win"] = df.groupby("race_id")["cal_p"].transform(lambda x: softmax_within(x.to_numpy(), bundle["p_win_softmax_temp"]))
    df["P_top3"] = df.groupby("race_id")["cal_p"].transform(lambda x: x/x.sum()*3.0 if x.sum()>0 else x)
    df = df.sort_values("P_win", ascending=False).reset_index(drop=True)
    df["pred_rank"] = range(1, len(df)+1)
    runs_r = runs[runs["race_id"]==race_id].copy()
    df = df.merge(runs_r[["umaban","horse_name","jockey_name","trainer_name","win_odds","popularity",
                          "race_name","grade","weather","going","finish_raw","finish_status","waku"]],
                  on="umaban", how="left", suffixes=("","_r"))
    for c in ["horse_name","jockey_name","trainer_name","win_odds","popularity","weather","going","waku"]:
        if c+"_r" in df.columns:
            df[c] = df[c].fillna(df[c+"_r"]) if c in df.columns else df[c+"_r"]
            df = df.drop(columns=[c+"_r"])

    horses = []
    for _, r in df.iterrows():
        horses.append({
            "rank": int(r["pred_rank"]),
            "umaban": int(r["umaban"]),
            "waku": int(r["waku"]) if pd.notna(r.get("waku")) else None,
            "horse_name": r.get("horse_name"),
            "jockey_name": r.get("jockey_name"),
            "trainer_name": r.get("trainer_name"),
            "P_win": float(r["P_win"]),
            "P_top3": float(r["P_top3"]),
            "win_odds": float(r["win_odds"]) if pd.notna(r.get("win_odds")) else None,
            "popularity": int(r["popularity"]) if pd.notna(r.get("popularity")) else None,
            "style_pref": r.get("style_pref"),
            "factors": {
                "same_show_rate": r.get("same_show_rate"),
                "recent_score": r.get("recent_score"),
                "jockey_win_rate": r.get("jockey_win_rate"),
                "trainer_win_rate": r.get("trainer_win_rate"),
                "course_win_rate": r.get("course_win_rate"),
                "horse_same_jc_win_rate": r.get("horse_same_jc_win_rate"),
                "past_trend_adj": r.get("past_trend_adj"),
                "body_weight": r.get("body_weight"),
                "body_weight_diff": r.get("body_weight_diff"),
                "last3f_recent": r.get("last3f_recent"),
                "position_avg": r.get("position_avg"),
                "kinryo_dev": r.get("kinryo_dev"),
                "n_prev": r.get("n_prev"),
            },
            "finish_num": float(r["finish_num"]) if pd.notna(r.get("finish_num")) else None,
            "finish_status": r.get("finish_status"),
        })
    axis = horses[0]; pb = horses[1] if len(horses)>1 else None; pc = horses[2] if len(horses)>2 else None
    candidates = []
    if axis["win_odds"] and axis["win_odds"] > 1.0:
        candidates.append({"role":"軸単","bet_type":"単勝","combination":str(axis["umaban"]),
                           "P_hit":axis["P_win"],"est_payout":axis["win_odds"]*100,"EV":axis["P_win"]*axis["win_odds"]})
        est_show = estimate_show_payout(axis["win_odds"])
        candidates.append({"role":"軸複","bet_type":"複勝","combination":str(axis["umaban"]),
                           "P_hit":axis["P_top3"],"est_payout":est_show,"EV":axis["P_top3"]*est_show/100.0})
        for label, pp in [("馬連2位", pb), ("馬連3位", pc)]:
            if pp is None or pp["win_odds"] is None: continue
            est = estimate_quinella_payout(axis["win_odds"], pp["win_odds"])
            ph = quinella_prob(axis["P_win"], pp["P_win"])
            candidates.append({"role":label,"bet_type":"馬連",
                "combination":f"{min(axis['umaban'], pp['umaban'])}-{max(axis['umaban'], pp['umaban'])}",
                "P_hit":ph,"est_payout":est,"EV":ph*est/100.0})
    info_row = df.iloc[0]
    info = {
        "race_id": race_id,
        "race_name": info_row.get("race_name") or "-",
        "date": info_row.get("date").strftime("%Y-%m-%d") if pd.notna(info_row.get("date")) else "-",
        "place": info_row.get("place"),
        "surface": info_row.get("surface"),
        "distance_m": int(info_row.get("distance_m")) if pd.notna(info_row.get("distance_m")) else None,
        "head_count": int(info_row.get("head_count")) if pd.notna(info_row.get("head_count")) else None,
        "weather": info_row.get("weather"),
        "going": info_row.get("going"),
        "pace": info_row.get("pace"),
        "bias_label": info_row.get("bias_label"),
        "decision_pattern": info_row.get("decision_pattern"),
        "race_shape": info_row.get("race_shape"),
        "grade": info_row.get("grade"),
        "circuit": info_row.get("circuit"),
    }
    return {"info":info, "top1_P_win":float(axis["P_win"]),
            "horses":horses, "candidates":candidates,
            "buy_recommended":[c for c in candidates if c["EV"]>=2.5]}


# ───────────────────────────────────────────────────────────
# UI ヘルパ
# ───────────────────────────────────────────────────────────

def confidence_label(p):
    if p >= 0.13:  return ("超高", "🔥", "#FF0844", "super")
    if p >= 0.10:  return ("高",   "✨", "#FF6B35", "high")
    if p >= 0.085: return ("中",   "⚖️", "#4A90E2", "mid")
    if p >= 0.075: return ("やや低", "🌫️", "#5C6370", "low")
    return ("低",   "❄️", "#2C3142", "cold")


def gauge_chart(p, title="自信度", height=240):
    """Plotly のゲージチャートで自信度を可視化。"""
    if p >= 0.13:    bar_color = "#FF0844"
    elif p >= 0.10:  bar_color = "#FF6B35"
    elif p >= 0.085: bar_color = "#4A90E2"
    else:            bar_color = "#5C6370"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=p * 100,
        number={"suffix": "%", "font": {"color": "#FAFAFA", "size": 36}},
        title={"text": title, "font": {"color": "#B0B7C3", "size": 14}},
        gauge={
            "axis": {"range": [0, 35], "tickwidth": 1, "tickcolor": "#B0B7C3",
                     "tickfont": {"color": "#B0B7C3"}},
            "bar": {"color": bar_color, "thickness": 0.6},
            "bgcolor": "rgba(255,255,255,0.05)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 7.5], "color": "rgba(92,99,112,0.2)"},
                {"range": [7.5, 8.5], "color": "rgba(92,99,112,0.4)"},
                {"range": [8.5, 10], "color": "rgba(74,144,226,0.4)"},
                {"range": [10, 13], "color": "rgba(255,107,53,0.4)"},
                {"range": [13, 35], "color": "rgba(255,8,68,0.4)"},
            ],
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def horse_ranking_chart(horses, top_n=12):
    """全頭の P_win 比較バーチャート。"""
    sub = horses[:top_n]
    names = [f"{h['umaban']}. {h.get('horse_name','-')}" for h in sub]
    p_wins = [h["P_win"] * 100 for h in sub]
    colors = [confidence_label(h["P_win"])[2] for h in sub]
    fig = go.Figure(go.Bar(
        x=p_wins, y=names, orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:.1f}%" for v in p_wins],
        textposition="outside",
        textfont=dict(color="#FAFAFA", size=12),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=max(280, 32*len(sub)),
        margin=dict(l=10, r=40, t=10, b=10),
        yaxis=dict(autorange="reversed", tickfont=dict(color="#FAFAFA", size=12), gridcolor="rgba(255,255,255,0.05)"),
        xaxis=dict(tickfont=dict(color="#B0B7C3"), gridcolor="rgba(255,255,255,0.05)", title=dict(text="P_win (%)", font=dict(color="#B0B7C3"))),
        showlegend=False,
    )
    return fig


def conf_distribution_chart(top1_df):
    """全レースの自信度分布ヒストグラム。"""
    p_wins = top1_df["P_win"].dropna() * 100
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=p_wins, nbinsx=20,
        marker=dict(color="#FF6B35", line=dict(color="#FFA552", width=1)),
        opacity=0.85,
    ))
    fig.add_vline(x=13, line_dash="dash", line_color="#FF0844", annotation_text="超高(13%)", annotation_position="top")
    fig.add_vline(x=10, line_dash="dash", line_color="#FF6B35", annotation_text="高(10%)", annotation_position="top")
    fig.add_vline(x=8.5, line_dash="dash", line_color="#4A90E2", annotation_text="中(8.5%)", annotation_position="top")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=260, margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(title=dict(text="自信度 (top1 P_win %)", font=dict(color="#B0B7C3")),
                   tickfont=dict(color="#B0B7C3"), gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(title=dict(text="レース数", font=dict(color="#B0B7C3")),
                   tickfont=dict(color="#B0B7C3"), gridcolor="rgba(255,255,255,0.05)"),
        showlegend=False,
    )
    return fig


def render_badge(text: str, kind: str = "high") -> str:
    return f'<span class="badge badge-{kind}">{text}</span>'


# ───────────────────────────────────────────────────────────
# Streamlit セットアップ
# ───────────────────────────────────────────────────────────

st.set_page_config(
    page_title="🏇 競馬予想 AI",
    page_icon="🏇",
    layout="wide",
    initial_sidebar_state="expanded",
)

# カスタムCSS 注入
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# session_state 初期化
if "selected_race_id" not in st.session_state:
    st.session_state["selected_race_id"] = ""


# ───────────────────────────────────────────────────────────
# サイドバー
# ───────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        '<div style="text-align:center; padding: 12px 0 8px;">'
        '<div style="font-size:2.5em; line-height:1;">🏇</div>'
        '<div style="font-size:1.4em; font-weight:800; color:#FFA552;">競馬予想 AI</div>'
        '<div style="font-size:0.8em; color:#7C8290;">v1.0 純フォーム機械学習</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.divider()

    meta = load_meta()

    mode = st.radio(
        "モード",
        ["🆕 これからのレース", "📅 過去日付の一覧", "🎯 個別レース予想"],
        index=0,
        label_visibility="collapsed",
    )

    st.divider()

    upcoming_date = None; show_upcoming = False
    date_input = None; show_list = False
    race_id_input = ""; predict_button = False

    if mode == "🆕 これからのレース":
        upcoming_dir = APP_DIR / "upcoming"
        upcoming_dates = []
        if upcoming_dir.exists():
            upcoming_dates = sorted(
                [d.name for d in upcoming_dir.iterdir() if d.is_dir() and (d/"predictions.parquet").exists()],
                reverse=True,
            )
        if upcoming_dates:
            upcoming_date = st.selectbox("日付（自動取得済）", upcoming_dates, index=0)
            show_upcoming = st.button("🚀 予想を表示", type="primary", use_container_width=True)
        else:
            st.warning("⚠️ 自動取得データがまだありません")
            st.caption("ターミナルで以下を実行:")
            st.code("python src/auto/fetch_upcoming.py --days tomorrow\npython src/auto/predict_upcoming.py --date YYYYMMDD", language="bash")

    elif mode == "📅 過去日付の一覧":
        feats = load_features()
        all_dates = sorted(feats["date"].dt.strftime("%Y-%m-%d").dropna().unique(), reverse=True)
        if all_dates:
            date_input = st.selectbox("日付", all_dates, index=0)
            show_list = st.button("📊 一覧を表示", type="primary", use_container_width=True)

    else:  # 個別レース予想
        race_id_input = st.text_input(
            "race_id",
            value=st.session_state.get("selected_race_id") or "202644011411",
            help="12桁のレースID",
        )
        predict_button = st.button("🎯 予想する", type="primary", use_container_width=True)

    st.divider()

    with st.expander("ℹ️ モデル情報"):
        st.markdown(f"""
**バージョン**: {meta['version']}
**学習期間**: {meta['trained_on']}
**特徴量数**: {meta['n_features']}
**ブレンド**: LightGBM 40% + CatBoost 30% + Ranker 30%
""")
        st.caption("Phase A: ターゲット見直し / B: 特徴量拡充 / C: アンサンブル / D: walk-forward 検証")

    st.divider()
    st.caption("🔄 自動取得: 朝7時・夜22時・毎時30分")


# ───────────────────────────────────────────────────────────
# メイン: ヒーロー
# ───────────────────────────────────────────────────────────

st.markdown('<div class="hero">競馬予想 AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">'
    '純フォーム機械学習 (LightGBM + CatBoost + LambdaRank アンサンブル) × Isotonic 校正で、'
    '<b>自信度の正確さ</b>を最大化したモデル。市場オッズに依存せず、馬体重・騎手特性・コース適性・近走指数など 34 特徴量からのみ判定。'
    '</div>',
    unsafe_allow_html=True,
)

# ───────────────────────────────────────────────────────────
# モード: 🆕 これからのレース
# ───────────────────────────────────────────────────────────
if mode == "🆕 これからのレース":
    if not show_upcoming or not upcoming_date:
        c1, c2, c3 = st.columns(3)
        c1.markdown("""<div class="race-card">
        <h3 style="margin-top:0;">🆕 当日 / 翌日のレース</h3>
        <p style="color:#B0B7C3;">自動取得済みの出馬表データで、JRA・NAR の全レース予想を一覧表示します。</p>
        </div>""", unsafe_allow_html=True)
        c2.markdown("""<div class="race-card">
        <h3 style="margin-top:0;">🔄 自動取得</h3>
        <p style="color:#B0B7C3;">launchd により<b>毎朝7時・毎夜22時・毎時30分</b>に出馬表とオッズを更新し、予想を生成します。</p>
        </div>""", unsafe_allow_html=True)
        c3.markdown("""<div class="race-card">
        <h3 style="margin-top:0;">🎯 自信度ランキング</h3>
        <p style="color:#B0B7C3;">全レースを top1 P_win 順に並べ、超高 🔥 / 高 ✨ / 中 ⚖️ で色分け表示。</p>
        </div>""", unsafe_allow_html=True)
        st.info("👈 サイドバーで自動取得済の日付を選んで「🚀 予想を表示」してください。")
        st.stop()

    up_dir = APP_DIR / "upcoming" / upcoming_date
    pred_df = pd.read_parquet(up_dir / "predictions.parquet")
    fdate = f"{upcoming_date[:4]}-{upcoming_date[4:6]}-{upcoming_date[6:8]}"
    top1 = pred_df[pred_df["pred_rank"] == 1].copy()

    # ── ヘッダ + サマリ ─────────────────────
    st.markdown(f"## 🆕 {fdate} のレース予想")
    places = sorted(pred_df["place"].unique().tolist())
    sel_places = st.multiselect("場で絞り込み", places, default=places)
    pdf = pred_df[pred_df["place"].isin(sel_places)].copy()
    top1 = pdf[pdf["pred_rank"] == 1].copy()

    cols = st.columns(5)
    cols[0].metric("📅 レース数", f"{top1['race_id'].nunique()}")
    cols[1].metric("🔥 超高(≥13%)", f"{(top1['P_win']>=0.13).sum()}")
    cols[2].metric("✨ 高(10-13%)", f"{((top1['P_win']>=0.10)&(top1['P_win']<0.13)).sum()}")
    cols[3].metric("⚖️ 中(8.5-10%)", f"{((top1['P_win']>=0.085)&(top1['P_win']<0.10)).sum()}")
    cols[4].metric("❄️ 低(<7.5%)", f"{(top1['P_win']<0.075).sum()}")

    # 自信度分布グラフ
    st.markdown("### 📊 自信度分布")
    st.plotly_chart(conf_distribution_chart(top1), use_container_width=True, config={"displayModeBar": False})

    st.divider()

    # ── 自信度ランキング表 ──────────────────
    st.markdown("### 🎯 自信度ランキング")
    list_show = top1.copy().sort_values("P_win", ascending=False)
    list_show["自信度"] = list_show["P_win"].apply(lambda v: f"{v*100:.1f}%")
    list_show["ランク"] = list_show["P_win"].apply(lambda v: f"{confidence_label(v)[1]} {confidence_label(v)[0]}")
    list_show["コース"] = list_show.apply(lambda r: f"{r.get('surface','')}{int(r.get('distance_m', 0))}m", axis=1)
    list_show["軸馬"] = list_show.apply(lambda r: f"{int(r['umaban'])}.{r['horse_name']}", axis=1)
    cols_pick = ["race_no","place","race_name","コース","head_count","pace","race_shape","自信度","ランク","軸馬","jockey_name","style_pref"]
    show = list_show[[c for c in cols_pick if c in list_show.columns]].rename(columns={
        "race_no":"R","place":"場","race_name":"レース名","head_count":"頭",
        "pace":"ペース","race_shape":"形態","jockey_name":"騎手","style_pref":"軸脚質",
    })
    st.dataframe(show, hide_index=True, use_container_width=True, height=min(560, 50 + 35 * len(show)))

    st.divider()

    # ── レース詳細 (タブ) ──────────────────
    st.markdown("### 🔍 個別レース詳細")
    options = [
        (f"R{int(r['race_no'])} {r['place']} {r['race_name']} ({r['P_win']*100:.1f}%)", r["race_id"])
        for _, r in top1.sort_values(["place", "race_no"]).iterrows()
    ]
    if options:
        labels = [o[0] for o in options]
        sel = st.selectbox("レースを選択", labels)
        sel_rid = next(o[1] for o in options if o[0] == sel)
        race_top = top1[top1["race_id"] == sel_rid].iloc[0]
        race_horses = pdf[pdf["race_id"] == sel_rid].sort_values("pred_rank")
        horses_list = [
            {"rank":int(r["pred_rank"]), "umaban":int(r["umaban"]), "horse_name":r.get("horse_name"),
             "jockey_name":r.get("jockey_name"), "trainer_name":r.get("trainer_name"),
             "P_win":float(r["P_win"]), "P_top3":float(r["P_top3"]), "style_pref":r.get("style_pref")}
            for _, r in race_horses.iterrows()
        ]

        col_g, col_info = st.columns([1, 2])
        with col_g:
            st.plotly_chart(gauge_chart(float(race_top["P_win"]), title="軸馬 自信度"),
                            use_container_width=True, config={"displayModeBar":False})
        with col_info:
            st.markdown(f"#### {race_top['race_name']}")
            ms = st.columns(4)
            ms[0].metric("場", race_top["place"])
            ms[1].metric("コース", f"{race_top['surface']}{int(race_top['distance_m'])}m")
            ms[2].metric("頭数", f"{int(race_top['head_count'])}")
            ms[3].metric("予想ペース", race_top["pace"])
            ms2 = st.columns(4)
            ms2[0].metric("形態", race_top["race_shape"])
            ms2[1].metric("軸馬", f"{int(race_top['umaban'])}. {race_top['horse_name']}")
            ms2[2].metric("騎手", race_top.get("jockey_name", "-"))
            ms2[3].metric("脚質", race_top.get("style_pref") or "-")

        st.markdown("##### 全頭 P_win 比較")
        st.plotly_chart(horse_ranking_chart(horses_list), use_container_width=True, config={"displayModeBar":False})

        with st.expander("📋 全頭詳細テーブル", expanded=False):
            h_show = race_horses[["pred_rank","umaban","horse_name","jockey_name","trainer_name","style_pref","P_win","P_top3","strength"]].copy()
            h_show["P_win"] = h_show["P_win"].apply(lambda v: f"{v*100:.1f}%")
            h_show["P_top3"] = h_show["P_top3"].apply(lambda v: f"{v*100:.1f}%")
            h_show["strength"] = h_show["strength"].apply(lambda v: f"{v:.3f}" if pd.notna(v) else "—")
            h_show.columns = ["順位","馬番","馬名","騎手","厩舎","予測脚質","P_win","P_top3","strength"]
            st.dataframe(h_show, hide_index=True, use_container_width=True)

    csv = pdf.to_csv(index=False).encode("utf-8-sig")
    st.download_button(f"📥 {fdate} 予想 CSV", csv,
                       file_name=f"predictions_{upcoming_date}.csv", mime="text/csv")
    st.stop()


# ───────────────────────────────────────────────────────────
# モード: 📅 過去日付の一覧
# ───────────────────────────────────────────────────────────
if mode == "📅 過去日付の一覧":
    if not show_list or not date_input:
        st.info("👈 サイドバーで日付を選択してください。")
        st.stop()

    bundle = load_model()
    feats = load_features()
    runs = load_runs()
    target = feats[feats["date"].dt.strftime("%Y-%m-%d") == date_input]
    if target.empty:
        st.warning(f"{date_input} の対象レースがありません")
        st.stop()

    NUMERIC_FEATS = bundle["numeric_feats"]; CATEGORICAL_FEATS = bundle["categorical_feats"]
    df = target.copy()
    for c in CATEGORICAL_FEATS:
        if c in df.columns: df[c] = df[c].astype("category")
    X = df[NUMERIC_FEATS + CATEGORICAL_FEATS]
    p_lgb = bundle["lgb_clf"].predict_proba(X)[:,1]
    X_cb = X.copy()
    for c in CATEGORICAL_FEATS:
        X_cb[c] = X_cb[c].astype(str).fillna("missing").replace({"nan":"missing","None":"missing"})
    p_cat = bundle["cb_clf"].predict_proba(X_cb)[:,1]
    rk_score = bundle["rk"].predict(X)
    df["rk_score"] = rk_score
    df["p_rk"] = df.groupby("race_id")["rk_score"].transform(lambda x: softmax_within(x.to_numpy(), bundle["rk_softmax_temp"]))
    df["p_lgb"]=p_lgb; df["p_cat"]=p_cat
    w = bundle["blend_weights"]
    df["p_blend"] = w["lgb"]*df["p_lgb"]+w["cat"]*df["p_cat"]+w["rk"]*(df["p_rk"]*3.0).clip(0,1)
    df["cal_p"] = bundle["iso"].transform(df["p_blend"])
    df["P_win"] = df.groupby("race_id")["cal_p"].transform(lambda x: softmax_within(x.to_numpy(), bundle["p_win_softmax_temp"]))
    top1 = df.sort_values("P_win", ascending=False).groupby("race_id").head(1).copy()
    top1 = top1.merge(runs[["race_id","umaban","horse_name","jockey_name","win_odds","popularity","race_name","grade"]],
                     on=["race_id","umaban"], how="left", suffixes=("","_r"))
    for c in ["horse_name","jockey_name","win_odds","popularity","race_name","grade"]:
        if c+"_r" in top1.columns:
            top1[c] = top1[c].fillna(top1[c+"_r"]) if c in top1.columns else top1[c+"_r"]
    top1["race_no"] = top1["race_id"].astype(str).str[-2:].astype(int)

    st.markdown(f"## 📅 {date_input} の全レース ({len(top1)} レース)")
    cols = st.columns(5)
    cols[0].metric("📅 レース数", f"{len(top1)}")
    cols[1].metric("🔥 超高", f"{(top1['P_win']>=0.13).sum()}")
    cols[2].metric("✨ 高", f"{((top1['P_win']>=0.10)&(top1['P_win']<0.13)).sum()}")
    cols[3].metric("⚖️ 中", f"{((top1['P_win']>=0.085)&(top1['P_win']<0.10)).sum()}")
    cols[4].metric("❄️ 低", f"{(top1['P_win']<0.075).sum()}")

    st.markdown("### 📊 自信度分布")
    st.plotly_chart(conf_distribution_chart(top1), use_container_width=True, config={"displayModeBar":False})

    st.markdown("### 🎯 自信度ランキング")
    ls = top1.copy().sort_values("P_win", ascending=False)
    ls["自信度"] = ls["P_win"].apply(lambda v: f"{v*100:.1f}%")
    ls["ランク"] = ls["P_win"].apply(lambda v: f"{confidence_label(v)[1]} {confidence_label(v)[0]}")
    ls["コース"] = ls.apply(lambda r: f"{r.get('surface','')}{int(r.get('distance_m', 0))}m", axis=1)
    ls["軸馬"] = ls.apply(lambda r: f"{int(r['umaban'])}.{r.get('horse_name','-')}", axis=1)
    ls["着順"] = ls["finish_num"].apply(lambda v: f"{int(v)}着" if pd.notna(v) else "—")
    show_cols = ["race_no","place","race_name","コース","head_count","pace","race_shape","自信度","ランク","軸馬","jockey_name","popularity","着順"]
    s = ls[[c for c in show_cols if c in ls.columns]].rename(columns={
        "race_no":"R","place":"場","race_name":"レース名","head_count":"頭",
        "pace":"ペース","race_shape":"形態","jockey_name":"騎手","popularity":"人気",
    })
    st.dataframe(s, hide_index=True, use_container_width=True, height=min(560, 50 + 35 * len(s)))
    st.stop()


# ───────────────────────────────────────────────────────────
# モード: 🎯 個別レース予想
# ───────────────────────────────────────────────────────────
if not predict_button:
    st.info("👈 サイドバーで race_id を入力してください。")
    c1, c2, c3 = st.columns(3)
    c1.markdown("""<div class="race-card">
    <h3 style="margin-top:0;">🎯 個別予想</h3>
    <p style="color:#B0B7C3;">race_id を指定して、軸4点（軸単・軸複・馬連2点）の EV を含む詳細予想を取得。</p>
    </div>""", unsafe_allow_html=True)
    c2.markdown("""<div class="race-card">
    <h3 style="margin-top:0;">📊 ゲージで一目</h3>
    <p style="color:#B0B7C3;">自信度をゲージチャートで可視化。13% 超なら 🔥、10% 超なら ✨。</p>
    </div>""", unsafe_allow_html=True)
    c3.markdown("""<div class="race-card">
    <h3 style="margin-top:0;">💎 EV ≥ 2.5</h3>
    <p style="color:#B0B7C3;">買い推奨（EV 2.5 以上）と見送りを明示。リスク管理も同時に。</p>
    </div>""", unsafe_allow_html=True)
    st.stop()

with st.spinner("予想計算中..."):
    bundle = load_model()
    feats = load_features()
    runs = load_runs()
    result = predict_race(race_id_input.strip(), bundle, feats, runs)

if "error" in result:
    st.error(result["error"])
    st.stop()

info = result["info"]

# ── レース見出し ──────────────────────────────
st.markdown(f"## 📋 {info.get('race_name','-')}")
st.caption(f"{info.get('place')} {info.get('surface')}{info.get('distance_m')}m | {info.get('head_count')}頭 | {info.get('date')}")
cols = st.columns(7)
cols[0].metric("場", info.get('place','-'))
cols[1].metric("距離", f"{info.get('surface','-')}{info.get('distance_m','-')}m")
cols[2].metric("頭数", f"{info.get('head_count','-')}頭")
cols[3].metric("グレード", info.get('grade') or "-")
cols[4].metric("馬場", info.get('going') or "-")
cols[5].metric("ペース", info.get('pace') or "-")
cols[6].metric("形態", info.get('race_shape') or "-")

st.divider()

# ── 自信度 (ゲージ + サマリ) ──────────────────
st.markdown("## 🎯 予想自信度")
top1_p = result["top1_P_win"]
conf_name, conf_emoji, conf_color, conf_kind = confidence_label(top1_p)

col_g, col_info = st.columns([1, 2])
with col_g:
    st.plotly_chart(gauge_chart(top1_p, title="top1 P_win"), use_container_width=True, config={"displayModeBar":False})

with col_info:
    axis = result["horses"][0]
    st.markdown(
        f'### {conf_emoji} {conf_name} 自信度  '
        f'<span class="badge badge-{conf_kind}">{top1_p*100:.1f}%</span>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f"**軸馬**: 馬番 **{axis['umaban']}** **{axis.get('horse_name','-')}**  \n"
        f"**騎手**: {axis.get('jockey_name','-')}  /  **厩舎**: {axis.get('trainer_name','-')}  \n"
        f"**予測脚質**: {axis.get('style_pref') or '-'}  /  **単勝オッズ**: {axis.get('win_odds','-')}  /  **人気**: {axis.get('popularity','-')}番"
    )

    # 推奨判定
    if top1_p >= 0.13:
        st.success("🎯 **超高自信度** — 軸単勝 + 馬連が狙い目（高的中率セグメント）")
    elif top1_p >= 0.10:
        st.info("✨ **高自信度** — 軸単勝の期待値が高い")
    elif top1_p >= 0.085:
        st.info("⚖️ **中自信度** — 軸複勝・馬連を中心に")
    elif top1_p >= 0.075:
        st.warning("🌫️ **やや低** — EV判定を厳しく")
    else:
        st.warning("❄️ **見送り推奨** — 精度が出にくいレース")

st.divider()

# ── 全頭ランキング (グラフ + テーブル タブ) ──────────────────
st.markdown("## 📊 全頭予想ランキング")
tab1, tab2 = st.tabs(["📊 グラフ", "📋 テーブル"])
with tab1:
    st.plotly_chart(horse_ranking_chart(result["horses"]), use_container_width=True, config={"displayModeBar":False})
with tab2:
    rk = pd.DataFrame([
        {
            "順位": h["rank"], "馬番": h["umaban"], "枠": h["waku"],
            "馬名": h.get("horse_name"), "騎手": h.get("jockey_name"),
            "厩舎": h.get("trainer_name"), "脚質": h.get("style_pref") or "-",
            "P_win": f"{h['P_win']*100:.1f}%", "P_top3": f"{h['P_top3']*100:.1f}%",
            "単勝オッズ": h["win_odds"], "人気": h["popularity"],
            "実着順": int(h["finish_num"]) if h.get("finish_num") and not pd.isna(h["finish_num"]) else None,
        }
        for h in result["horses"]
    ])
    st.dataframe(rk, hide_index=True, use_container_width=True)

st.divider()

# ── 軸4点候補 + EV ──────────────────────────────
st.markdown("## 🎲 軸4点候補")
if result["candidates"]:
    cand_df = pd.DataFrame(result["candidates"])
    cand_df["推奨"] = cand_df["EV"].apply(lambda v: "🟢 買い" if v >= 2.5 else "見送り")
    cand_show = cand_df[["role","bet_type","combination","P_hit","est_payout","EV","推奨"]].rename(columns={
        "role":"役割","bet_type":"券種","combination":"組合せ","P_hit":"的中率","est_payout":"推定払戻"
    })
    cand_show["的中率"] = cand_show["的中率"].apply(lambda v: f"{v*100:.1f}%")
    cand_show["推定払戻"] = cand_show["推定払戻"].apply(lambda v: f"{v:,.0f}円")
    cand_show["EV"] = cand_show["EV"].apply(lambda v: f"{v:.2f}")
    st.dataframe(cand_show, hide_index=True, use_container_width=True)

    if result["buy_recommended"]:
        st.success(f"### 💎 買い推奨 {len(result['buy_recommended'])} 点")
        for c in result["buy_recommended"]:
            st.markdown(f"- **{c['role']}**: {c['bet_type']} `{c['combination']}` ・ EV **{c['EV']:.2f}** ・ 的中率 {c['P_hit']*100:.1f}%")
    else:
        st.warning("⚠️ 見送り推奨：EV ≥ 2.5 を満たす候補なし")

st.divider()

# ── 詳細特徴量 ───────────────────────────────
st.markdown("## 📖 各馬の予想詳細")
with st.expander("🐴 全頭の要素分解スコア", expanded=False):
    detail = pd.DataFrame([
        {
            "順位": h["rank"], "馬番": h["umaban"], "馬名": h.get("horse_name"),
            "脚質": h.get("style_pref") or "-", "P_win": h["P_win"],
            "馬体重": h["factors"].get("body_weight"),
            "増減": h["factors"].get("body_weight_diff"),
            "同条件複勝率": h["factors"].get("same_show_rate"),
            "騎手単勝率": h["factors"].get("jockey_win_rate"),
            "馬×騎手×コース勝率": h["factors"].get("horse_same_jc_win_rate"),
            "調教師勝率": h["factors"].get("trainer_win_rate"),
            "コース勝率": h["factors"].get("course_win_rate"),
            "近走指数": h["factors"].get("recent_score"),
            "上がり3F": h["factors"].get("last3f_recent"),
            "平均通過順位": h["factors"].get("position_avg"),
            "斤量差": h["factors"].get("kinryo_dev"),
            "過去傾向補正": h["factors"].get("past_trend_adj"),
            "出走数": h["factors"].get("n_prev"),
        } for h in result["horses"]
    ])
    detail["P_win"] = detail["P_win"].apply(lambda v: f"{v*100:.1f}%")
    for c in ["同条件複勝率","騎手単勝率","馬×騎手×コース勝率","調教師勝率","コース勝率"]:
        detail[c] = detail[c].apply(lambda v: f"{v*100:.1f}%" if pd.notna(v) else "—")
    st.dataframe(detail, hide_index=True, use_container_width=True)

# 実結果 (過去レース)
finished = [h for h in result["horses"] if h.get("finish_num") and h.get("finish_status")=="ok"]
if finished:
    st.divider()
    st.markdown("## 🏁 実際の結果")
    actual_1st = next((h for h in finished if h["finish_num"]==1), None)
    if actual_1st:
        pred_rank = actual_1st["rank"]
        if pred_rank == 1:
            st.success(f"### 🎯 軸馬の単勝 **的中**!  予想1位 = 実1着 (馬番 {actual_1st['umaban']} {actual_1st['horse_name']})")
        elif pred_rank <= 3:
            st.info(f"### 📊 1着馬は予想 **{pred_rank}位** (馬番 {actual_1st['umaban']} {actual_1st['horse_name']})")
        else:
            st.warning(f"### ❌ 1着馬は予想 **{pred_rank}位** (馬番 {actual_1st['umaban']} {actual_1st['horse_name']})")
    res = pd.DataFrame([
        {"着順":int(h["finish_num"]), "馬番":h["umaban"], "馬名":h.get("horse_name"),
         "予想順位":h["rank"], "P_win":f"{h['P_win']*100:.1f}%",
         "オッズ":h.get("win_odds"), "人気":h.get("popularity")}
        for h in finished
    ]).sort_values("着順").head(5)
    st.dataframe(res, hide_index=True, use_container_width=True)
