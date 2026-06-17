# 🏇 競馬予想 v1.0

純フォーム機械学習 (市場オッズ非依存) で「自信度の正確さ」を最大化した競馬予想 Web アプリ。

## 機能

- **race_id 入力** → 完成版モデル (Phase C アンサンブル) で予想
- **自信度判定** (top1 P_win) + ランク表示
- **全頭ランキング** (P_win / P_top3 / オッズ / 人気 / 実着順)
- **軸4点候補と EV** + 買い推奨/見送り判定
- **各馬の予想詳細** (馬体重・騎手特性・厩舎・コース勝率・近走指数 など 13 要素)
- **実結果との照合** (過去レースのみ)

## モデル仕様

- **アンサンブル**: LightGBM (40%) + CatBoost (30%) + LambdaRank (30%)
- **校正**: Isotonic Regression で生確率を実勝率にマッピング
- **特徴量**: 34 個 (市場オッズ・人気は使わない)
  - 同条件複勝率・近走指数・脚質×展開・騎手×場×距離帯・厩舎×場
  - 馬体重・増減 (直近5走平均)・上がり3F・通過順位
  - レース内 相対特徴量 (馬間差)
  - 馬場バイアス・予測ペース・形態判定

## 性能

| 指標 | 値 |
|---|---:|
| キャリブレーション誤差 (最大) | ±0.6〜1.2pp |
| top1 of 18頭立て 1着的中率 | 17.2% (2026 年) |
| top3 of 18頭立て 3着以内含む率 | 60.7% |
| Walk-Forward 100% 超セグメント数 | 3〜9 個 / 期間 |

## ローカル起動

```bash
# 1. 依存パッケージインストール
pip install -r requirements.txt

# 2. 起動
streamlit run app.py
```

ブラウザが自動で開いて `http://localhost:8501` でアクセスできます。

## ディレクトリ構成

```
keiba_app/
├── app.py              # Streamlit アプリ本体
├── requirements.txt    # 依存パッケージ
├── README.md           # このファイル
├── model/
│   ├── keiba_v1.pkl       # 学習済みモデル (1.9 MB)
│   └── keiba_v1_meta.json # メタデータ
└── data/
    ├── features_v2.parquet  # 特徴量 (~10 MB)
    ├── runs_jra.parquet     # JRA 出走データ
    ├── runs_nar.parquet     # NAR 出走データ
    ├── payouts_jra.parquet  # JRA 払戻
    └── payouts_nar.parquet  # NAR 払戻
```

データセット: 2024-01〜2026-06 の JRA + NAR (約 16 万出走)

## デプロイ

### Streamlit Cloud (無料、推奨)

1. このディレクトリ (`keiba_app`) を GitHub に push
2. https://share.streamlit.io で sign in
3. 「New app」 → リポジトリ選択 → `app.py` を指定 → Deploy

### Hugging Face Spaces

1. https://huggingface.co/spaces で「Create new Space」
2. Space SDK: **Streamlit** 選択
3. リポジトリに `app.py` と `requirements.txt`, `model/`, `data/` を push

### Render / Railway / Fly.io

`requirements.txt` + `app.py` を含むリポジトリを連携 → Web Service 作成 →
起動コマンド: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`

## race_id サンプル

データセットに含まれる過去レース ID なら何でも予想可能。

| race_id | 場 | 日 | レース |
|---|---|---|---|
| 202644011411 | 川崎 | 2026-01-14 | 14R |
| 202610010304 | 小倉 | 2026-01-31 | (障害=対象外) |
| 202644020111 | 大井 | 2026-02-01 | 1R |

## 注意事項

- **過去レースの予想・答え合わせ**用です。未来レース予想には別途出馬表データの取得が必要。
- **新馬戦・障害戦は学習対象外** (過去走無し or 別カテゴリのため)。
- 予想は **市場オッズを参照しません** (純フォームのみ)。EV計算と買い推奨判定でのみ単勝オッズを使用。
- **投資は自己責任** で。バックテスト ROI ≠ 実運用 ROI。
