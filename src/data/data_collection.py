import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import yaml
from logger import get_logger




logger = get_logger("DVC loge Datacollecation")
logger.info("Trying to Loading dataset...")
data=pd.read_csv("water_potability.csv")
logger.info("Loading dataset Completed")
#print(data.head)
logger.info("performing trin test split ...")
train_data ,test_data =train_test_split(data,test_size=0.20,random_state=42)
logger.info("completed trin test split ...")

logger.info("creating  path ")
data_path=os.path.join("data","row",)

os.makedirs(data_path,exist_ok =True)
logger.info("path creation completed.")

logger.info("trying to write the data in to csv...")
train_data.to_csv(os.path.join(data_path,"train.csv"),index=False)
test_data.to_csv(os.path.join(data_path,"test.csv"),index=False)
logger.info("Completed to load the csv data ...")