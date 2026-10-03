# Natural Language Processing 自然語言處理

自然語言處理課程的四次作業與期末專題紀錄，涵蓋詞向量、文本分類、命名實體辨識、LLM 工具呼叫與自動作文評分。

## 目錄結構

- `nlp_as1_B1228005/`：詞向量與詞語類比
- `nlp_as2_B1228005/`：Transformer 新聞分類
- `nlp_as3_B1228005/`：生醫命名實體辨識
- `nlp_as4_B1228005/`：LLM Function Calling
- `Final Project/`：期末專題，自動作文評分

## 作業內容

| 作業 | 主題 | 資料集 | 使用技術 |
| --- | --- | --- | --- |
| 作業一 | 詞向量與詞語類比 | Google Word Analogy、英文維基百科 | GloVe、Word2Vec、Gensim、t-SNE |
| 作業二 | 新聞文本分類 | AG News | PyTorch、Transformer Encoder、BERT Tokenizer |
| 作業三 | 生醫命名實體辨識 | NCBI Disease | BERT、Hugging Face Transformers、seqeval |
| 作業四 | LLM 工具呼叫 | 本地模擬股價、天氣與新聞資料 | Groq、Qwen、Llama、JSON、Python 工具路由 |

## 檔案用途

| 檔案 | 用途 |
| --- | --- |
| `nlp_as1_B1228005.ipynb` | 比較預訓練 GloVe 與自行訓練 Word2Vec 的詞語類比表現，並視覺化詞向量。 |
| `nlp_as2_B1228005.ipynb` | 建立 Transformer Encoder，將 AG News 新聞分成四類並評估分類結果。 |
| `nlp_as3_B1228005.ipynb` | 微調 BERT，辨識生醫文本中的疾病名稱並計算 NER 評估指標。 |
| `nlp_as4_B1228005.ipynb` | 實作 LLM 判斷工具、查詢模擬資料庫、生成回答的流程。 |
| 各作業的 `.docx`／`.pdf` | 記錄實作過程、實驗結果與分析。 |
| `requirement.txt`／`requirements.txt` | 記錄各作業使用的套件與版本。 |
| `fc_utils.py` | 提供 LLM 呼叫、JSON 解析、資料查詢及測試函式。 |
| `mock_data.json` | 提供工具呼叫使用的模擬股票、天氣與新聞資料，非即時資訊。 |
| `fc_log.json` | 保存 Qwen 模型的工具呼叫測試結果。 |
| `fc_log_llama-3.3-70b-versatile.json` | 保存 Llama 模型的工具呼叫測試結果，供模型比較。 |
| 作業三、四的 `.zip` | 保存對應作業的壓縮提交檔案。 |

## 期末專題

**自動作文評分**（Learning Agency Lab — Automated Essay Scoring 2.0）

- **目標**：根據學生作文內容，預測 1 至 6 分的整體評分。
- **資料集**：AES 2.0 競賽資料、PERSUADE 2.0 語料。
- **使用技術**：TF-IDF、作文特徵工程、LightGBM、Longformer、ELECTRA、集成模型。

| 檔案 | 用途 |
| --- | --- |
| `NLP_team2_checkpoint4.ipynb` | 進行資料前處理、特徵擷取、模型訓練、集成與提交檔案產生。 |
| `NLP_team2_checkpoint4.odt` | 期末專題報告，記錄方法與實驗成果。 |
| `README.md` | 期末專題的模型、環境、執行方式與成果說明。 |

## 環境需求

主要工具為 Python、Jupyter、Gensim、PyTorch、Hugging Face Transformers、Keras/KerasNLP、LightGBM、scikit-learn、spaCy、NLTK 與 Groq。

各作業的套件版本請參考資料夾內的環境紀錄。期末專題主要在 Kaggle 執行；作業四需自行設定 `GROQ_API_KEY`。
