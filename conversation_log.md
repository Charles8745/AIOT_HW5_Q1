# AI / Human 文章偵測器 - 開發對話記錄

## 專案資訊
- **專案名稱**: AIOT_HW5_Q1
- **GitHub**: https://github.com/Charles8745/AIOT_HW5_Q1
- **Demo Site**: https://aiothw5q1-pwbdgcqrxsjwxpzxhqvegq.streamlit.app/
- **日期**: 2025年12月7日

---

## 📋 開發過程摘要

### 階段一：專案初始建立

根據用戶提供的設計規格，建立了完整的 AI/Human 文章偵測器專案結構：

**建立的檔案：**
- `app.py` - Streamlit 主程式
- `requirements.txt` - Python 依賴套件
- `README.md` - 專案說明文件
- `src/features.py` - 特徵提取模組
- `src/models.py` - ML 模型 (TF-IDF, RandomForest, Ensemble)
- `src/transformer_detector.py` - Transformer 偵測器
- `src/groq_client.py` - Groq API 客戶端
- `src/utils.py` - 工具函數
- `src/__init__.py` - 套件初始化
- `data/samples.csv` - 訓練資料集 (30 筆樣本)

**環境設置：**
```powershell
# 建立虛擬環境
python -m venv venv

# 啟動虛擬環境
.\venv\Scripts\Activate.ps1

# 安裝依賴
pip install -r requirements.txt

# 啟動應用程式
streamlit run app.py
```

---

### 階段二：四項優化功能

用戶選擇了以下 4 項優化：

#### 1. 擴充訓練資料集
- 從 30 筆擴充至 **120+ 筆樣本**
- 新增 `language` 欄位 (en/zh)
- 60+ 英文樣本 + 60+ 中文樣本

#### 2. 使用真實 Transformer 模型
- 整合 `roberta-base-openai-detector` (OpenAI 官方 AI 偵測模型)
- 建立 `RealTransformerDetector` 類別
- 支援多種預訓練模型切換

#### 3. 新增 Perplexity 特徵
- 建立 `src/perplexity.py` 模組
- 使用 GPT-2 計算文本困惑度
- AI 生成文本通常困惑度較低
- 包含 `PerplexityCalculator` 和 `BurstinessCalculator` 類別

#### 4. 新增中文支援
- 安裝 jieba 中文分詞套件
- 更新 `features.py` 支援雙語特徵提取
- 建立 `BilingualFeatureExtractor` 類別
- 中文虛詞列表支援

```powershell
# 安裝 jieba
pip install jieba
```

---

### 階段三：暗色模式

新增暗色/亮色主題切換功能：

**實作內容：**
- 在 `app.py` 新增 `get_theme_css()` 函數
- 新增 `apply_dark_theme_to_fig()` 函數為 Plotly 圖表套用主題
- 側邊欄新增主題切換開關
- 使用 `st.session_state` 保持主題狀態

**UI 文字更新：**
```python
UI_TEXT = {
    'en': {
        'theme': '🎨 Theme',
        'dark_mode': '🌙 Dark Mode',
        'light_mode': '☀️ Light Mode'
    },
    'zh': {
        'theme': '🎨 主題',
        'dark_mode': '🌙 深色模式',
        'light_mode': '☀️ 淺色模式'
    }
}
```

---

### 階段四：檔案整理與上傳 GitHub

#### 建立 .gitignore
```
# Virtual Environment
venv/
env/
.venv/

# Python cache
__pycache__/
*.py[cod]

# IDE
.idea/
.vscode/

# Environment variables
.env

# Model cache
.cache/
models/
```

#### 刪除快取資料夾
```powershell
Remove-Item -Recurse -Force "src\__pycache__"
```

#### 更新 README.md
- 完整專案說明
- 系統架構圖
- 安裝步驟
- 功能介紹
- 依賴套件列表

#### Git 操作
```powershell
# 添加所有檔案
git add .

# 提交
git commit -m "feat: AI/Human Text Detector - Complete Implementation"

# 推送到 GitHub
git push origin main
```

---

### 階段五：調整專案結構

將 `ai_detector/` 資料夾內的檔案移至根目錄：

```powershell
# 移動檔案
Move-Item -Path "ai_detector\*" -Destination "." -Force

# 刪除空資料夾
Remove-Item -Path "ai_detector" -Recurse -Force

# 提交更改
git add -A
git commit -m "refactor: Move files to root directory"
git push origin main
```

---

### 階段六：新增 Demo Site 連結

在 README.md 開頭新增 Demo Site 連結：

```markdown
## 🌐 Demo Site

👉 **[點擊這裡體驗線上 Demo](https://aiothw5q1-pwbdgcqrxsjwxpzxhqvegq.streamlit.app/)**
```

```powershell
git add README.md
git commit -m "docs: Add demo site link"
git push origin main
```

---

## 📁 最終專案結構

```
AIOT_HW5_Q1/
├── .gitattributes
├── .gitignore
├── README.md
├── app.py                      # Streamlit 主程式 (980+ 行)
├── requirements.txt
├── data/
│   └── samples.csv             # 120+ 訓練樣本
├── src/
│   ├── __init__.py
│   ├── features.py             # 雙語特徵提取
│   ├── models.py               # ML 模型
│   ├── transformer_detector.py # Transformer 偵測器
│   ├── perplexity.py           # 困惑度計算
│   ├── groq_client.py          # Groq API
│   └── utils.py                # 工具函數
└── venv/                       # 虛擬環境 (不上傳)
```

---

## ✨ 功能總覽

| 功能 | 說明 |
|------|------|
| 🎯 二分類結果 | Human / AI 分類判斷 |
| 📊 AI 生成機率 | 各模型的信心分數 (0-100%) |
| 🤝 多模型投票 | Ensemble Decision 整合判斷 |
| 🔬 可解釋特徵 | 20+ 統計/語言特徵與量化視覺化 |
| 🌐 雙語支援 | 支援英文與中文文章偵測 |
| 📈 困惑度分析 | Perplexity 特徵輔助判斷 |
| 🌙 暗色模式 | 可切換亮色/暗色主題 |
| 🤖 Groq API | 即時生成 AI 文本測試 |

---

## 🔧 技術棧

| 類別 | 技術 |
|------|------|
| Web Framework | Streamlit |
| ML | scikit-learn (TF-IDF, RandomForest) |
| Deep Learning | PyTorch, Transformers (RoBERTa) |
| NLP | jieba (中文分詞) |
| Visualization | Plotly, Matplotlib |
| API | Groq |

---

## 📝 Git Commits 記錄

1. `feat: AI/Human Text Detector - Complete Implementation` - 12 files, 3506 insertions
2. `refactor: Move files to root directory` - 12 files renamed
3. `docs: Add demo site link` - 1 file, 6 insertions

---

## 🔗 相關連結

- **GitHub Repository**: https://github.com/Charles8745/AIOT_HW5_Q1
- **Demo Site**: https://aiothw5q1-pwbdgcqrxsjwxpzxhqvegq.streamlit.app/
