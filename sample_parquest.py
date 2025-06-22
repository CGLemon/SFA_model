import pandas as pd
import numpy as np
import difflib
import os, glob, logging
import time, argparse
import csv
import random

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Config:
    def __init__(self):
        self.maxsize = 500
        self.sample_rate_per_chunk = 0.001
        self.max_check_urls = 500

def get_unique_urls(df):
    url_table = df["url"].copy()
    url_table = set(url_table)
    url_table = list(url_table)
    url_table = sorted(url_table)
    return url_table

def lcs_difflib(s1, s2):
    seq_matcher = difflib.SequenceMatcher(None, s1, s2)
    match = seq_matcher.find_longest_match(0, len(s1), 0, len(s2))
    return s1[match.a: match.a + match.size], match.size

def remove_common_elem(substring):
    common = [
        "https://",
        ".com.tw/",
        ".com/",
        ".com",
        "www."
    ]
    for elem in common:
        substring = substring.replace(elem, "")
    return substring, len(substring)

def random_sample(df, urls_buf, config):
    df_url = get_unique_urls(df)

    sample_urls_idx = np.random.choice(
        len(df_url), round(len(df_url) * config.sample_rate_per_chunk), replace=False)
    check_urls_idx = np.random.choice(
        len(urls_buf), min(len(urls_buf), config.max_check_urls), replace=False)
    num_accpet = 0
    num_refuse = 0

    for sample_idx in sample_urls_idx:
        if len(urls_buf) >= config.maxsize:
            break
        sample_url = df_url[sample_idx]
        highest_match_rate = 0.0
        highest_match_url = str()
        highest_match_sub = str()
        for url_idx in check_urls_idx:
            url = urls_buf[url_idx]

            substring, match_size = lcs_difflib(sample_url, url)
            substring, match_size = remove_common_elem(substring)
            basestring_size = min(len(sample_url), len(url))

            if highest_match_rate < match_size / basestring_size:
                highest_match_rate = match_size / basestring_size
                highest_match_url = url
                highest_match_sub = substring
        if (1.0 - highest_match_rate) > np.random.uniform(0.0, 1.0):
            urls_buf.append(sample_url)
            num_accpet += 1
        else:
            num_refuse += 1
    return num_accpet, num_refuse


def main():
    parser = argparse.ArgumentParser(description="URL Sampling Tool")
    parser.add_argument("-i", "--input-path", metavar="<PATH>" ,required=True, help="Directory containing input Parquet files")
    parser.add_argument("-o", "--output-path", metavar="<PATH>", required=True, help="Output CSV path to save sampled URLs")
    parser.add_argument("-s", "--maxsize", type=int, metavar="<INT>", required=True, help="Maximum number of URLs to sample")
    parser.add_argument("--sample-rate", type=float, metavar="<FLOAT>", default=0.001, help="Sampling rate per chunk")
    parser.add_argument("--max-check-urls", type=int, metavar="<INT>", default=500, help="Maximum number of URLs to check for similarity")
    args = parser.parse_args()

    config = Config()
    config.maxsize = args.maxsize
    config.sample_rate_per_chunk = args.sample_rate
    config.max_check_urls = args.max_check_urls

    urls_buf = list()
    all_accpet, all_refuse = 0, 0
    start_time = time.time()

    all_parquet = glob.glob(os.path.join(args.input_path, "**", "*.parquet"), recursive=True)
    random.shuffle(all_parquet)
    for idx, filename in enumerate(all_parquet):
        try:
            df = pd.read_parquet(filename)
            logger.info("[{:<2}] Loaded file: {:<40}".format(idx + 1, filename))

            num_accpet, num_refuse = random_sample(df, urls_buf, config)
            accpet_rate = num_accpet / (num_accpet + num_refuse)
            elapsed = time.time() - start_time
            logger.info("     ├─ Buffer size  : {:>6} items".format(len(urls_buf)))
            logger.info("     ├─ Accept rate  : {:>6.2f} %".format(accpet_rate * 100.0))
            logger.info("     └─ Elapsed time : {:>6.2f} sec".format(elapsed))

            all_accpet += num_accpet
            all_refuse += num_refuse
        except Exception as e:
            logger.error(f"Error processing file {filename}: {e}")
            continue
        if len(urls_buf) >= config.maxsize:
            break

    logger.info("\n=== Summary ===")
    logger.info("Total accepted rate : {:>6.2f} %".format(all_accpet / (all_accpet + all_refuse) * 100.0))
    logger.info("Total Elapsed time : {:>6.2f} sec".format(time.time() - start_time))

    pd.DataFrame({"url": urls_buf, "label": 0}).to_csv(
        args.output_path,
        index=False,
        quotechar='"',
        quoting=csv.QUOTE_ALL
    )
    logger.info("Saved {} URLs to {}".format(len(urls_buf), args.output_path))

if __name__ == "__main__":
    main()
