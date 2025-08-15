import pandas as pd
import numpy as np
import os

from logger import get_logger


logger = get_logger("DVC log transformation")
logger.info("Loading train and test dataset...")
train_data = pd.read_csv(r"data\row\train.csv")
test_data = pd.read_csv(r"data\row\test.csv")
logger.info("completed teain and test Loading dataset...")

logger.info("fill ing the missing valus")
def fill_missing_with_mean(df):
    try:
        for column in df.columns:
            if df[column].isnull().any():
                median_value = df[column].mean()
                df[column].fillna(median_value,inplace=True)
        return df
    except Exception as e:
        raise Exception(f"Error Filling missing values with mean:{e}")
    
logger.info("complited missing valus")


train_process_data=fill_missing_with_mean(train_data)
test_process_data=fill_missing_with_mean(test_data)

data_path=os.path.join("data","process")

os.makedirs(data_path,exist_ok=True)

logger.info("writting the row data to process data")

train_process_data.to_csv(os.path.join(data_path,"train_process_data.csv"),index=False)
test_process_data.to_csv(os.path.join(data_path,"test_process_data.csv"),index=False)

logger.info("completed the row data to process data")