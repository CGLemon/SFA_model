import os
import sys
import logging
import pandas as pd
import numpy as np
import joblib
import argparse
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, f1_score

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def load_model(model_path):
    """加載保存的模型和分類器"""
    try:
        logger.info(f"正在從 {model_path} 加載模型...")
        model_data = joblib.load(model_path)
        return model_data
    except Exception as e:
        logger.error(f"加載模型時出錯: {str(e)}")
        raise

def predict(model_data, test_path, output_path=None):
    """使用加載的模型進行預測"""
    try:
        model = model_data['model']
        classifier = model_data['classifier']
        categories = model_data['categories']
        
        # 加載測試數據
        logger.info(f"正在加載測試數據: {test_path}")
        test_data = pd.read_csv(test_path)
        
        # 檢查測試集是否有標籤
        has_label = 'label' in test_data.columns
        
        # 準備特徵
        X_test = test_data.drop(['label', 'url'], axis=1) if has_label else test_data.drop(['url'], axis=1)
        y_test = test_data['label'] if has_label else None
        
        # 進行預測
        logger.info("正在進行預測...")
        test_preds = classifier.predict_proba(model, (X_test, y_test) if y_test is not None else (X_test, None))
        
        # 獲取正類的機率（第二列）
        test_preds_pos = test_preds[:, 1]
        test_preds_binary = (test_preds_pos > 0.5).astype(int)
        
        # 保存預測結果
        if output_path:
            result_df = pd.DataFrame({
                'url': test_data['url'],
                'prob_0': 1 - test_preds_pos,
                'prob_1': test_preds_pos,
                'prediction': test_preds_binary
            })
            result_df.to_csv(output_path, index=False)
            logger.info(f"預測結果已保存至: {output_path}")
        
        # 如果有標籤，計算評估指標
        if has_label and y_test is not None:
            logger.info("\n=== 測試集評估 ===")
            logger.info(f"測試集AUC: {roc_auc_score(y_test, test_preds_pos):.4f}")
            logger.info(f"測試集準確率: {accuracy_score(y_test, test_preds_binary):.4f}")
            logger.info(f"測試集F1分數: {f1_score(y_test, test_preds_binary):.4f}")
            logger.info("\n測試集分類報告:")
            logger.info(f"\n{classification_report(y_test, test_preds_binary)}")
        
        return test_preds_pos, test_preds_binary
    
    except Exception as e:
        logger.error(f"預測時出錯: {str(e)}", exc_info=True)
        raise

def main():
    # 解析參數
    parser = argparse.ArgumentParser(description='使用已訓練的SFA LightGBM模型進行預測')
    parser.add_argument('--test_path', type=str, required=True, help='測試集CSV文件路徑')
    parser.add_argument('--model_path', type=str, default='saved_models/sfa_lgbm_model.joblib', 
                       help='模型文件路徑 (默認: saved_models/sfa_lgbm_model.joblib)')
    parser.add_argument('--output_path', type=str, default='predictions.csv', 
                       help='預測結果輸出路徑 (默認: predictions.csv)')
    
    args = parser.parse_args()
    
    try:
        # 加載模型
        model_data = load_model(args.model_path)
        
        # 進行預測
        predict(model_data, args.test_path, args.output_path)
        
        logger.info("\n=== 預測完成 ===")
        
    except Exception as e:
        logger.error(f"程序執行出錯: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
