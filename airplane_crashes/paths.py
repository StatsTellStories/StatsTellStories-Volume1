from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "dataset"

''' RAW DATA '''
RAW_DIR = DATA_DIR / "raw"

''' ADSB LOL'''
TRACES_DIR = RAW_DIR / "adsblol"
TRACES_NO_ACCIDENT = TRACES_DIR / "no_crash"
TRACES_CRASH = TRACES_DIR / "crash"

''' ASN '''
ASN_DIR = RAW_DIR / "asn"

''' KAGGLE '''
KAGGLE_DIR = RAW_DIR / "Kaggle_Crashes_1908"

''' World Bank '''
WB_DIR = RAW_DIR / "world_bank"

''' EXPORTED DATA '''
EXPORT_DIR = DATA_DIR / "exports"


''' PARQUET TRACES '''
PARQUET_DIR = DATA_DIR / "parquet_traces"
# CRASH_DATES_DIR = PARQUET_DIR / "dates_crashes"
PARQUET_TRACES_NO_CRASH = PARQUET_DIR /"no_crash"
PARQUET_TRACES_CRASH = PARQUET_DIR / "crash"
PARQUET_TRACES_ALL = PARQUET_DIR / "all"

''' PROCESSED '''
PROCESSED_DIR = DATA_DIR / "processed"

''' PRE-PROCESSED '''
PRE_PROCESSED_DIR = DATA_DIR / "preprocessed"

''' Exported Images '''
IMAGE_DIR = ROOT / "exported_images"
#KAGGLE_IMAGES = IMAGE_DIR / "2_kaggle_analysis"
DESC_ANALYSIS_IMAGES = IMAGE_DIR / "3_descriptive_analysis"
