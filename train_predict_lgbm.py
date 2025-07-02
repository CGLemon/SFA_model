import os
import sys
import time
import logging
import pandas as pd
import numpy as np
import argparse
import joblib
from tqdm import tqdm
from pathlib import Path
from SFALGBMClassifier import SFALGBMClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix, fbeta_score

# 設置日誌
def setup_logging(debug=False):
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('training.log')
        ]
    )
    return logging.getLogger(__name__)

def parse_args():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(description='使用SFA的LightGBM實現訓練模型和進行預測')
    subparsers = parser.add_subparsers(dest='command', help='子命令', required=True)
    
    # 訓練命令
    train_parser = subparsers.add_parser('train', help='訓練模型')
    train_parser.add_argument('--train_path', type=str, required=True, help='訓練集CSV文件路徑')
    train_parser.add_argument('--val_path', type=str, help='驗證集CSV文件路徑（可選）')
    train_parser.add_argument('--model_dir', type=str, default='saved_models', help='模型保存目錄')
    train_parser.add_argument('--seed', type=int, default=42, help='隨機種子')
    train_parser.add_argument('--n_trials', type=int, default=1, help='Optuna優化試驗次數')
    train_parser.add_argument('--use_gpu', action='store_true', help='是否使用GPU進行訓練')
    train_parser.add_argument('--debug', action='store_true', help='啟用調試模式')
    
    # 預測命令
    predict_parser = subparsers.add_parser('predict', help='使用訓練好的模型進行預測')
    predict_parser.add_argument('--test_path', type=str, required=True, help='測試集CSV文件路徑')
    predict_parser.add_argument('--output_path', type=str, default='predictions.csv', help='預測結果輸出路徑')
    predict_parser.add_argument('--model_path', type=str, default='saved_models/sfa_lgbm_model.joblib', 
                              help='模型文件路徑')
    predict_parser.add_argument('--debug', action='store_true', help='啟用調試模式')
    
    return parser.parse_args()

def load_data(train_path, val_path=None, test_path=None, seed=42, val_size=0.2):
    """加載和準備數據
    
    參數:
        train_path: 訓練集路徑
        val_path: 驗證集路徑，如果為None則從訓練集中分割
        test_path: 測試集路徑
        seed: 隨機種子
        val_size: 驗證集比例（當val_path為None時使用）
    """
    logger = logging.getLogger(__name__)
    logger.info("=== 開始加載數據 ===")
    
    try:
        # 加載訓練集
        logger.info(f"加載訓練集: {train_path}")
        train_df = pd.read_csv(train_path)
        
        # 移除 url 列（如果存在）
        if 'url' in train_df.columns:
            logger.info("移除 'url' 列，因為 LightGBM 需要數值型特徵")
            train_df = train_df.drop(columns=['url'])
        
        # 如果沒有提供驗證集，則從訓練集中分割
        if val_path is None:
            logger.info(f"未提供驗證集，將從訓練集中分割 {val_size*100:.0f}% 作為驗證集")
            from sklearn.model_selection import train_test_split
            train_df, val_df = train_test_split(
                train_df, 
                test_size=val_size, 
                random_state=seed,
                stratify=train_df['label'] if 'label' in train_df.columns else None
            )
            logger.info(f"訓練集大小: {len(train_df):,}, 驗證集大小: {len(val_df):,}")
            
            # 處理訓練集
            X_train = train_df.drop(columns=['label'] if 'label' in train_df.columns else [])
            y_train = train_df['label'].values if 'label' in train_df.columns else None
            
            # 處理驗證集
            X_val = val_df.drop(columns=['label'] if 'label' in val_df.columns else [])
            y_val = val_df['label'].values if 'label' in val_df.columns else None
        else:
            # 加載驗證集
            logger.info(f"加載驗證集: {val_path}")
            val_df = pd.read_csv(val_path)
            
            # 移除 url 列（如果存在）
            if 'url' in val_df.columns:
                logger.info("移除驗證集中的 'url' 列")
                val_df = val_df.drop(columns=['url'])
            
            # 處理訓練集
            X_train = train_df.drop(columns=['label'] if 'label' in train_df.columns else [])
            y_train = train_df['label'].values if 'label' in train_df.columns else None
            
            # 處理驗證集
            X_val = val_df.drop(columns=['label'] if 'label' in val_df.columns else [])
            y_val = val_df['label'].values if 'label' in val_df.columns else None
        
        # 加載測試集
        if test_path:
            logger.info(f"加載測試集: {test_path}")
            test_df = pd.read_csv(test_path)
            
            # 移除 url 列（如果存在）
            if 'url' in test_df.columns:
                test_df = test_df.drop(columns=['url'])
            
            X_test = test_df.drop(columns=['label'] if 'label' in test_df.columns else [])
            y_test = test_df['label'].values if 'label' in test_df.columns else None
        else:
            X_test, y_test = None, None
        
        # 獲取類別信息
        categories = list(X_train.columns)
        
        # 數據集詳情
        ds_details = {
            'name': 'custom_dataset',
            'n_features': len(categories),
            'n_samples': len(X_train)
        }
        
        logger.info(f"訓練集大小: {len(X_train):,}, 特徵數: {len(categories)}")
        if X_val is not None:
            logger.info(f"驗證集大小: {len(X_val):,}")
        if X_test is not None:
            logger.info(f"測試集大小: {len(X_test):,}")
        
        # 返回數據集
        return (X_train, y_train, X_val, y_val, X_test, y_test, categories, ds_details)
        
    except Exception as e:
        logger.error(f"加載數據時出錯: {str(e)}")
        raise

def train_model(args):
    """訓練模型並保存"""
    global logger
    logger = setup_logging(args.debug)
    logger.info("=== 開始訓練模型 ===")
    
    try:
        # 加載數據
        X_train, y_train, X_val, y_val, X_test, y_test, categories, ds_details = load_data(
            train_path=args.train_path,
            val_path=args.val_path,
            test_path=None,
            seed=args.seed
        )
        
        # 設置模型目錄
        model_dir = Path(args.model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / 'sfa_lgbm_model.joblib'
        
        # 初始化分類器
        # ds_details 需要包含 (ds_name, num_samples, num_features, num_classes, class_dist)
        ds_info = (
            ds_details['name'],  # 數據集名稱
            ds_details['n_samples'],  # 樣本數量
            ds_details['n_features'],  # 特徵數量
            2,  # 假設是二分類
            None  # 類別分佈，可以為 None
        )
        lgbm_clf = SFALGBMClassifier(ds_info, args.seed)
        
        # 設置設備
        if args.use_gpu:
            lgbm_clf.device = 'gpu'
            logger.info("使用GPU進行訓練")
        else:
            lgbm_clf.device = 'cpu'
            logger.info("使用CPU進行訓練")
        
        # 超參數優化
        if args.n_trials > 0:
            logger.info(f"=== 開始超參數優化，試驗次數: {args.n_trials} ===")
            
            # 確保數據是 pandas DataFrame/Series
            import pandas as pd
            if not isinstance(X_train, pd.DataFrame):
                X_train_df = pd.DataFrame(X_train)
                X_val_df = pd.DataFrame(X_val) if X_val is not None else None
                y_train_s = pd.Series(y_train)
                y_val_s = pd.Series(y_val) if y_val is not None else None
            else:
                X_train_df = X_train
                X_val_df = X_val
                y_train_s = y_train
                y_val_s = y_val
            
            # 獲取類別信息
            categories = list(X_train_df.columns)
            
            # 設置數據集詳情
            ds_info = (ds_details['name'], len(X_train_df), len(categories), 2, None)  # 假設是二分類
            
            # 初始化分類器
            lgbm_clf = SFALGBMClassifier(ds_info, args.seed)
            
            # 運行優化
            num_trials, best_trial = lgbm_clf.run_optimization(
                X_train=X_train_df,
                y_train=y_train_s,
                X_test=X_val_df if X_val_df is not None else X_train_df,
                y_test=y_val_s if y_val_s is not None else y_train_s,
                categories=categories,
                n_trials=args.n_trials
            )
            
            best_params = best_trial.params if hasattr(best_trial, 'params') else {}
            logger.info(f"最佳參數: {best_params}")
            lgbm_clf.set_hyper_params(best_params)
        
        # 訓練最終模型
        logger.info("\n=== 訓練最終模型 ===")
        model = lgbm_clf.train(X_train, y_train)
        
        # 保存模型
        model_data = {
            'model': model,
            'classifier': lgbm_clf,
            'categories': categories,
            'ds_details': ds_details
        }
        joblib.dump(model_data, model_path)
        logger.info(f"模型已保存到 {model_path}")
        
        # 驗證集評估
        if X_val is not None and y_val is not None:
            logger.info("\n=== 驗證集評估 ===")
            val_preds = lgbm_clf.predict_proba(model, (X_val, y_val))
            val_preds_pos = val_preds[:, 1]
            val_preds_binary = (val_preds_pos > 0.5).astype(int)
            
            logger.info(f"驗證集AUC: {roc_auc_score(y_val, val_preds_pos):.4f}")
            logger.info(f"驗證集準確率: {accuracy_score(y_val, val_preds_binary):.4f}")
            logger.info(f"驗證集F1分數: {f1_score(y_val, val_preds_binary):.4f}")
        
        return model_path
        
    except Exception as e:
        logger.error(f"訓練過程中出錯: {str(e)}")
        raise

def predict(args):
    """使用訓練好的模型進行預測"""
    global logger
    logger = setup_logging(args.debug)
    logger.info("=== 開始預測 ===")
    
    try:
        # 加載模型
        logger.info(f"從 {args.model_path} 加載模型...")
        model_data = joblib.load(args.model_path)
        model = model_data['model']
        lgbm_clf = model_data['classifier']
        
        # 加載測試數據
        logger.info(f"加載測試數據: {args.test_path}")
        test_df = pd.read_csv(args.test_path)
        
        # 檢查是否包含URL列
        if 'url' not in test_df.columns:
            raise ValueError("測試數據必須包含'url'列")
            
        # 保存URL列用於最終輸出
        urls = test_df['url'].values
        
        # 獲取模型訓練時使用的特徵順序
        feature_names = model.feature_name() if hasattr(model, 'feature_name') else \
                      model.booster_.feature_name() if hasattr(model, 'booster_') else None
        
        if feature_names is None:
            # 如果無法獲取特徵名稱，則使用測試數據中的特徵（排除url和label）
            feature_columns = [col for col in test_df.columns if col not in ['url', 'label']]
            logger.warning(f"無法從模型獲取特徵名稱，使用測試數據中的特徵: {feature_columns}")
        else:
            feature_columns = feature_names
            logger.info(f"使用模型訓練時的特徵: {feature_columns}")
            
            # 檢查測試數據是否包含所有需要的特徵
            missing_features = set(feature_columns) - set(test_df.columns)
            if missing_features:
                raise ValueError(f"測試數據缺少以下特徵: {missing_features}")
        
        # 選擇特徵並確保順序一致
        X_test = test_df[feature_columns].copy()
        
        # 記錄特徵信息
        logger.info(f"測試數據特徵數量: {len(feature_columns)}")
        logger.info(f"測試數據特徵: {feature_columns}")
        logger.info(f"測試數據形狀: {X_test.shape}")
        
        # 檢查是否有缺失值
        if X_test.isnull().any().any():
            logger.warning("測試數據中包含缺失值，將使用0填充")
            X_test = X_test.fillna(0)
        
        # 進行預測
        logger.info("正在進行預測...")
        test_preds = lgbm_clf.predict_proba(model, (X_test, None))
        test_preds_pos = test_preds[:, 1]
        
        # 創建結果DataFrame
        results_df = pd.DataFrame({
            'url': urls,
            'prob_normal': 1 - test_preds_pos,
            'prob_malicious': test_preds_pos,
            'prediction': (test_preds_pos > 0.5).astype(int)
        })
        
        # 保存結果
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df[['url', 'prob_normal', 'prob_malicious', 'prediction']].to_csv(
            output_path, index=False, float_format='%.6f'
        )
        logger.info(f"預測結果已保存至: {output_path}")
        
        # 顯示統計信息
        logger.info("\n=== 預測結果統計 ===")
        logger.info(f"預測為惡意URL數量 (1): {results_df['prediction'].sum():,}")
        logger.info(f"預測為正常URL數量 (0): {(results_df['prediction'] == 0).sum():,}")
        logger.info(f"平均惡意URL預測概率: {results_df['prob_malicious'].mean():.6f}")
        
        # 顯示預測結果示例
        logger.info("\n=== 預測結果示例 ===")
        logger.info(f"\n{results_df.head().to_string()}")
        
        # 如果有真實標籤，計算評估指標
        if 'label' in test_df.columns and not test_df['label'].isna().all():
            y_test = test_df['label'].values
            logger.info("\n=== 測試集評估 ===")
            test_auc = roc_auc_score(y_test, test_preds_pos)
            test_acc = accuracy_score(y_test, results_df['prediction'])
            test_f1 = f1_score(y_test, results_df['prediction'])
            test_fbeta = fbeta_score(y_test, results_df['prediction'], beta=0.5)
            
            logger.info(f"測試集AUC: {test_auc:.4f}")
            logger.info(f"測試集準確率: {test_acc:.4f}")
            logger.info(f"測試集F1分數: {test_f1:.4f}")
            logger.info(f"測試集F0.5分數: {test_fbeta:.4f}")

            # 混淆矩陣
            cm = confusion_matrix(y_test, results_df['prediction'])
            cm_df = pd.DataFrame(
                cm,
                index=["實際:正常 (0)", "實際:異常 (1)"],
                columns=["預測:正常 (0)", "預測:異常 (1)"]
            )
            logger.info("\n混淆矩陣：\n" + str(cm_df))
        else:
            logger.info("未找到有效的真實標籤，跳過測試集評估")
    
    except Exception as e:
        logger.error(f"預測過程中出錯: {str(e)}")
        raise

def main():
    try:
        # 解析命令行參數
        args = parse_args()
        
        # 設置日誌級別
        global logger
        logger = setup_logging(args.debug)
        
        # 記錄開始時間
        start_time = time.time()
        
        # 執行相應的命令
        if args.command == 'train':
            logger.info("=== 開始訓練流程 ===")
            model_path = train_model(args)
            logger.info(f"訓練完成，模型已保存至: {model_path}")
            
        elif args.command == 'predict':
            logger.info("=== 開始預測流程 ===")
            predict(args)
            
        # 計算並記錄總執行時間
        elapsed_time = time.time() - start_time
        logger.info(f"\n=== 執行完成，總耗時: {elapsed_time:.2f} 秒 ===")
        
    except Exception as e:
        logger.error(f"程序執行出錯: {str(e)}", exc_info=args.debug)
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("\n用戶中斷程序執行")
        sys.exit(0)

if __name__ == "__main__":
    main()
