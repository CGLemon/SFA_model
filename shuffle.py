import pandas as pd
import numpy as np
import argparse, os, logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_file(input_files, output_file, ratio, max_samples_per_files):
    try:
        df_list = list()
        for filename in input_files:
            df = pd.read_csv(filename)
            rows, _ = df.shape
            if max_samples_per_files is not None and \
                max_samples_per_files < rows:
                df = df.sample(n=max_samples_per_files)
                logger.info(f"從 {filename} 抽取的原始數據形狀: {df.shape}")
            else:
                logger.info(f"{filename} 的原始數據形狀: {df.shape}")
            df_list.append(df)

        df = pd.concat(df_list)

        logger.info(f"合併後的原始數據形狀: {df.shape}")

        if ratio is None:
            df.to_csv(f"{output_file}", index=False)
        else:
            ratio = min(1.0, ratio)
            ratio = max(0.0, ratio)
            train_df = df.sample(n=round(rows * ratio))
            test_df = df.drop(train_df.index)
            
            logger.info(f"訓練集的數量: {train_df.shape[0]}")
            logger.info(f"測試集的數量: {test_df.shape[0]}")
            
            basename = os.path.basename(output_file)
            path = output_file[:-len(basename)]

            train_df.to_csv(f"{path}train_{basename}", index=False)
            test_df.to_csv(f"{path}test_{basename}", index=False)

    except Exception as e:
        logger.error(f"處理文件時出錯: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='URL 特徵提取工具')
    parser.add_argument('-i', '--input', type=str, nargs='+', required=True, metavar='<PATH>', help='輸入文件路徑')
    parser.add_argument('-o', '--output', type=str, required=True, metavar='<PATH>', help='輸出文件路徑')
    parser.add_argument('-r', '--ratio', type=float, metavar='<0.0 ~ 1.0>', help='訓練集分割的比例')
    parser.add_argument('--max-samples-per-files', type=int, metavar='<INT>', help='從每個 CSV 文件抽出最多 K 筆資料')
    args = parser.parse_args()
    
    try:
        process_file(args.input, args.output, args.ratio, args.max_samples_per_files)
    except Exception as e:
        logger.error(f"程序執行出錯: {str(e)}")
        return 1
    return 0

if __name__ == "__main__":
    main()
