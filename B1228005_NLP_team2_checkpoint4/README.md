# Learning Agency Lab — Automated Essay Scoring 2.0

**Team:** 胡樂麒 (B1228005)、羅立安 (B1228025)、黃柏碩 (B1228030)、蔡勇濱 (B1228039)  
**課程：** 自然語言處理  
**最終成績：** Public QWK = 0.80631 | Private QWK = 0.81372  
**Kaggle 競賽：** [Learning Agency Lab - Automated Essay Scoring 2.0](https://www.kaggle.com/c/learning-agency-lab-automated-essay-scoring-2)

---

## 專案架構

```
aes_submission/
├── ensemble_aes.ipynb      # 主要訓練與推論 notebook（Kaggle 環境執行）
├── upload_to_hf.py         # 上傳模型權重至 HuggingFace Hub
└── README.md               # 本文件
```

> **注意：** 模型權重檔案（`.pt`、`.h5`）不包含在本壓縮檔中，請至 HuggingFace 下載（見下方連結）。

---

## 模型架構總覽

| 模型 | 框架 | Public QWK | Private QWK |
|------|------|-----------|------------|
| LGBM + TF-IDF + 手工特徵 | LightGBM | 0.79506 | 0.80488 |
| Longformer-base-4096 | HuggingFace / PyTorch | 0.77328 | 0.79661 |
| ELECTRA-large-discriminator | KerasNLP / KerasHub | 0.75016 | 0.78383 |
| **Ensemble（LGBM+Longformer+ELECTRA）** | — | **0.80631** | **0.81372** |

---

## 環境需求

### Python 版本
```
Python 3.10+（建議使用 Kaggle Notebook 環境）
```

### 套件安裝

```bash
pip install keras-nlp keras tensorflow torch transformers \
            lightgbm scikit-learn pandas numpy scipy \
            spacy nltk matplotlib
```

```bash
# spaCy 英文模型
python -m spacy download en_core_web_sm

# NLTK 詞庫
python -c "import nltk; nltk.download('words')"
```


---

## 資料集準備（Kaggle 環境）

在 Kaggle Notebook 的 **Add Input** 面板中加入以下資料集：

| 資料集 | 用途 |
|--------|------|
| [AES 2.0 Competition Data](https://www.kaggle.com/c/learning-agency-lab-automated-essay-scoring-2/data) | 競賽訓練/測試資料 |
| [PERSUADE 2.0 Corpus](https://www.kaggle.com/datasets/nbroad/persaude-corpus-2) | 資料增強（約 25,000 篇） |
| [NLTK Data](https://www.kaggle.com/datasets/nltkdata/nltk-data) | 英文詞庫（拼字特徵） |

模型 Input（在 Kaggle Add Input → Models 搜尋加入）：
- `allenai/longformer-base-4096`（HuggingFace PyTorch 格式）
- `electra_large_discriminator_uncased_en`（KerasNLP preset）

---

## 執行步驟

### 在 Kaggle Notebook 執行

1. 上傳 `ensemble_aes.ipynb` 至 Kaggle Notebook
2. 加入上述資料集與模型 Input
3. 設定 GPU 加速器（建議 T4 x2 或 P100）
4. 修改 `RUN_MODELS` 字典選擇要跑的模型：

```python
RUN_MODELS = {
    "lgbm": True,          # LGBM + TF-IDF
    "bert": False,
    "roberta": False,
    "deberta_v3_small": False,
    "longformer": True,    # HuggingFace Longformer
    "electra": True,       # KerasNLP ELECTRA
    "albert": False,
}
```

5. 依序執行所有 cell，最後在 `/kaggle/working/submission.csv` 取得預測結果

## 模型權重下載

訓練完成的模型權重已上傳至 HuggingFace Hub：

```python
from huggingface_hub import hf_hub_download

# Longformer 權重
hf_hub_download(
    repo_id="Bin0982/aes2-models",
    filename="best_longformer.pt",
    local_dir="./weights"
)
# ELECTRA 權重
hf_hub_download(
    repo_id="Bin0982/aes2-models",
    filename="best_electra.weights.h5",
    local_dir="./weights"
)
```

---

## 技術細節

### 資料前處理
- 縮寫展開（20 個常見英文縮寫）
- 移除 HTML 標記、URL、多餘空白
- 頭尾截斷（前 256 字 + 後 256 字）供 Transformer 使用

### 特徵工程（供 LGBM 使用）
- 結構特徵：字數、段落數、句子數、平均句長
- 詞彙特徵：唯一詞數、詞彙豐富度（TTR）
- 拼字特徵：拼字錯誤數、拼字錯誤率（spaCy + NLTK）

### Ordinal Regression
將分數（1–6）轉換為 5 維二元標籤，使用 Binary Crossentropy 訓練，預測時對 sigmoid 輸出 > 0.5 的維度加總得到最終整數分數。

### Ensemble 搜尋策略
- 粗粒度搜尋（步長 0.05）：窮舉所有 1~3 模型組合
- 細粒度精煉（步長 0.01，radius=0.08）：對前 8 名組合再次搜尋
- 最終最佳權重：LGBM 0.32、Longformer 0.42、ELECTRA 0.26

---

## 參考資料

1. [KerasNLP AES 2.0 Starter Notebook](https://www.kaggle.com/code/awsaf49/aes-2-0-kerasnlp-starter)
2. [AES2 Deberta+LGBM+CountVectorizer [LB.819]](https://www.kaggle.com/code/hideyukizushi/aes2-deberta-lgbm-countvectorizer-lb-819)
3. [PERSUADE 2.0 Dataset](https://www.kaggle.com/datasets/nbroad/persaude-corpus-2)
4. [NLTK English Word Frequency Dataset](https://www.kaggle.com/datasets/rtatman/english-word-frequency)
