import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_custom_data(train_path, test_path, test_size=0.2, random_state=42):
    """
    加載自定義訓練集和測試集
    
    參數:
    - train_path: 訓練集CSV文件路徑
    - test_path: 測試集CSV文件路徑
    - test_size: 驗證集比例
    - random_state: 隨機種子
    
    返回:
    - X_train: 訓練集特徵
    - X_val: 驗證集特徵
    - X_test: 測試集特徵
    - y_train: 訓練集標籤
    - y_val: 驗證集標籤
    - y_test: 測試集標籤 (如果測試集有標籤)
    - categories: 分類特徵索引
    """
    # 加載訓練集
    train_data = pd.read_csv(train_path)
    
    # 分離特徵和標籤
    X_train_full = train_data.drop(['label', 'url'], axis=1)
    y_train_full = train_data['label']
    
    # 分割訓練集和驗證集
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y_train_full
    )
    
    # 加載測試集
    test_data = pd.read_csv(test_path)
    
    # 確保測試集特徵與訓練集一致
    X_test = test_data[X_train.columns]
    
    # 檢查測試集是否有標籤
    y_test = test_data['label'] if 'label' in test_data.columns else None
    
    # 識別分類特徵
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns
    categories = [i for i, col in enumerate(X_train.columns) if col in categorical_cols]
    
    # 準備完整的 ds_details
    ds_name = os.path.basename(train_path).replace('.csv', '')
    num_samples = len(X_train) + len(X_val)
    num_features = X_train.shape[1]
    num_classes = len(y_train.unique())
    class_dist = y_train.value_counts().to_dict()
    
    ds_details = (ds_name, num_samples, num_features, num_classes, class_dist)
    
    return X_train, X_val, X_test, y_train, y_val, y_test, categories, ds_details
