import pandas as pd
import numpy as np
import re
import os
import argparse
from urllib.parse import urlparse
from tqdm import tqdm
import logging
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_url_features(row):
    """從 URL 中提取特徵"""
    try:
        url = row['url']
        label = row['label']

        # 確保 URL 有 scheme
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
            
        parsed = urlparse(url)
        
        # 基本特徵
        url_netloc_len = len(parsed.netloc) if parsed.netloc else 0
        url_path_len = len(parsed.path) if parsed.path else 0
        url_count_dot = url.count('.')
        url_count_hyphen = url.count('-')
        
        # 域名特徵
        domain = parsed.netloc.split(':')[0] if parsed.netloc else ''
        domain_parts = domain.split('.')
        domain_len = len(domain)
        subdomain_count_dot = domain.count('.')
        tld = domain_parts[-1] if domain_parts else ''
        tld_len = len(tld)
        domain_count_hyphen = domain.count('-')
        domain_count_digit = sum(c.isdigit() for c in domain)
        
        # 路徑特徵
        path = parsed.path
        path_len = len(path)
        path_count_no_of_dir = path.count('/')
        path_count_zero = path.count('0')
        path_count_lower = sum(c.islower() for c in path)
        
        # 敏感詞檢查
        sensitive_words = ['admin', 'login', 'wp-content', 'include', 'shell', 'passwd', 'wp-', 'config']
        path_has_any_sensitive_words = int(any(word in path.lower() for word in sensitive_words))
        
        return {
            'url': url,
            'url_netloc_len': url_netloc_len,
            'url_path_len': url_path_len,
            'url_count_dot': url_count_dot,
            'url_count_hyphen': url_count_hyphen,
            'domain_len': domain_len,
            'subdomain_count_dot': subdomain_count_dot,
            'tld_len': tld_len,
            'domain_count_hyphen': domain_count_hyphen,
            'domain_count_digit': domain_count_digit,
            'path_len': path_len,
            'path_count_no_of_dir': path_count_no_of_dir,
            'path_count_zero': path_count_zero,
            'path_count_lower': path_count_lower,
            'path_has_any_sensitive_words': path_has_any_sensitive_words,
            'label': label
        }
    except Exception as e:
        logger.error(f"處理 URL 時出錯: {url}, 錯誤: {str(e)}")
        return None

def process_file(input_files, output_file, max_samples_per_files):
    """處理輸入文件並提取特徵"""
    logger.info(f"正在處理檔案: {input_files}")
    
    # 讀取數據
    try:
        df_list = list()
        for filename in input_files:
            df = pd.read_csv(filename)
            rows, _ = df.shape
            if max_samples_per_files is not None and \
                max_samples_per_files < rows:
                df = df.sample(n=max_samples_per_files)
            df_list.append(df)
        df = pd.concat(df_list)

        logger.info(f"原始數據形狀: {df.shape}")
        
        # 確保有 url 列
        if 'url' not in df.columns:
            # 如果沒有 url 列，假設第一列是 URL
            df = df.rename(columns={df.columns[0]: 'url'})
            logger.info(f"將第一列重命名為 'url'")
        
        # 提取特徵
        features = []
        for idx, _ in enumerate(tqdm(df['url'], desc="提取特徵")):
            row = df.iloc[idx]
            feature = extract_url_features(row)
            if feature:
                features.append(feature)
        
        if not features:
            raise ValueError("未能從數據中提取任何特徵")
            
        # 轉換為 DataFrame
        feature_df = pd.DataFrame(features)

        rows, columns = df.shape
        logger.info(f"提取特徵數量: {columns}")
        logger.info(f"保存欄位數量: {rows}")

        # 保存結果
        feature_df.to_csv(output_file, index=False)
        logger.info(f"特徵提取完成，已保存至: {output_file}")
        
        return feature_df
        
    except Exception as e:
        logger.error(f"處理文件時出錯: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='URL 特徵提取工具')
    parser.add_argument('-i', '--input', type=str, nargs='+', required=True, metavar='<PATH>', help='輸入文件路徑')
    parser.add_argument('-o', '--output', type=str, required=True, metavar='<PATH>', help='輸出文件路徑')
    parser.add_argument('--max-samples-per-files', type=int, metavar='<INT>', help='從每個 CSV 文件抽出最多 K 筆資料')
    args = parser.parse_args()
    
    try:
        process_file(args.input, args.output, args.max_samples_per_files)
    except Exception as e:
        logger.error(f"程序執行出錯: {str(e)}")
        return 1
    return 0

if __name__ == "__main__":
    main()
