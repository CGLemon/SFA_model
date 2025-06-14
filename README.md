# 惡意連結偵測模型

這個專案提供了一個用於偵測惡意連結的模型。它包含用於模型訓練、從 URL 提取特徵以及進行預測的腳本。

## 專案結構

* `train_predict_lgbm.py`: 使用 LightGBM 模型進行訓練和預測的腳本。
* `feature_extraction.py`: 用於從 URL 資料集中提取特徵的腳本。

## 前製準備

### 1. 下載程式

    $ git clone https://github.com/deenar-de/SFA_model
    $ cd SFA_model

### 2. 準備資料集

專案使用的訓練資料集來自 [Kaggle](https://www.kaggle.com/datasets/pilarpieiro/tabular-dataset-ready-for-malicious-url-detection)，該資料集總共包含 1,682,213 筆資料 ，分別為 ```train_dataset.csv``` 和 ```test_dataset.csv```，請下載後解壓縮到本目錄上。


## 使用方式

### 1. 特徵提取

在訓練或預測之前，您需要從 URL 資料中提取特徵。```feature_extraction.py``` 腳本負責此任務。它可以處理包含 URL 的 CSV 檔案，請執行


    python3 feature_extraction.py --input train_dataset.csv --output train_features.csv
    python3 feature_extraction.py --input test_dataset.csv --output test_features.csv

### 2. 訓練模型

執行訓練腳本時，程式將會自動隨機選擇並測試不同的超參數組合來進行模型訓練。```--n_trials``` 參數用於指定超參數測試的次數（例如，設定為 100 次）。完成測試後，程式將根據表現最佳的超參數組合，訓練出最終模型。訓練的模型將保存在 ```saved_models``` 底下。

    python3 train_predict_lgbm.py train --train_path train_features.csv --model_dir saved_models --n_trials 100

### 3. 驗證模型

最後載入訓練好的模型驗證其性能。

    python3 train_predict_lgbm.py predict --test_path test_features.csv --output_path predictions.csv --model_path saved_models/sfa_lgbm_model.joblib
