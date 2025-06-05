import pandas as pd
import os

# 文件路径
INPUT_DIR = "/home/rhyswei/Code/aiyeast/get_model/input/20250601_data"
WT_GENE_XLSX = os.path.join(INPUT_DIR, "WT_gene_filled.xlsx")
DATA_WT_XLSX = os.path.join(INPUT_DIR, "data_WT_filled.xlsx")
WT_GENE_CSV = os.path.join(INPUT_DIR, "WT_gene_filled.csv")
DATA_WT_CSV = os.path.join(INPUT_DIR, "data_WT_filled.csv")

def convert_xlsx_to_csv(xlsx_path, csv_path):
    print(f"正在转换 {xlsx_path} 到 {csv_path}...")
    try:
        df = pd.read_excel(xlsx_path)
        df.to_csv(csv_path, index=False)
        print(f"成功转换 {xlsx_path}")
    except FileNotFoundError:
        print(f"错误：未找到文件 {xlsx_path}")
    except Exception as e:
        print(f"转换文件时发生错误 {xlsx_path}: {e}")

if __name__ == "__main__":
    convert_xlsx_to_csv(WT_GENE_XLSX, WT_GENE_CSV)
    convert_xlsx_to_csv(DATA_WT_XLSX, DATA_WT_CSV)
    print("转换完成。")
 