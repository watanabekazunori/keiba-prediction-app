# 🚀 Streamlit Cloud デプロイ手順

## 1. リポジトリ準備 ✅ 完了済み

GitHub repo: **https://github.com/watanabekazunori/keiba-prediction-app**

## 2. Streamlit Cloud で公開

### ステップ 1: サインイン
[https://share.streamlit.io/](https://share.streamlit.io/) にアクセスして **「Continue with GitHub」** でサインイン。

### ステップ 2: New app

1. 右上 **「New app」** をクリック
2. 以下を入力：
   - **Repository**: `watanabekazunori/keiba-prediction-app`
   - **Branch**: `main`
   - **Main file path**: `app.py`
   - **App URL** (任意): `keiba-ai` など → `https://keiba-ai.streamlit.app/`
3. **「Deploy!」** をクリック

### ステップ 3: デプロイ完了を待つ（3〜5 分）

初回ビルドで `requirements.txt` の依存パッケージ（LightGBM・CatBoost・Plotly 等）が
インストールされます。デプロイ完了後、自動で URL が発行されます。

### 完成 🎉

`https://<your-app-name>.streamlit.app/` でアクセス可能になります。

---

## アップデート方法

ローカルでコードを修正したら：
```bash
cd keiba_app
git add .
git commit -m "Update message"
git push
```

→ Streamlit Cloud が **自動で再デプロイ**します（1〜2 分）。

---

## トラブルシューティング

### デプロイ失敗

- **「Manage app」** → **「Logs」** でエラー確認
- 依存パッケージのバージョン不整合: `requirements.txt` を見直し

### モデルが重い (LightGBM + CatBoost)

- 初回ロードに 30 秒〜 1 分かかります（`@st.cache_resource` でキャッシュされ 2回目以降は高速）

### データサイズ制限

- Streamlit Cloud: リポジトリ **1 GB まで**
- 現状の総サイズ: **34 MB** (余裕あり)

---

## 別の公開方法

### Hugging Face Spaces (無料)

1. [https://huggingface.co/spaces](https://huggingface.co/spaces) で **「Create new Space」**
2. Space SDK: **Streamlit**
3. ローカルの `keiba_app` を Space repo に push

### Render (無料枠あり)

1. [https://render.com](https://render.com) でこのリポジトリと接続
2. Service type: **Web Service**
3. Build: `pip install -r requirements.txt`
4. Start: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`
