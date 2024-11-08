import shutil
import os

"""
이 코드는 최종적으로 복구된 CSV 파일을 data/final/pipeline2 폴더로 이동시키는 코드입니다.
"""


# source path
source_path = 'data/preprocessed/clean_data_v2.csv'

# destination folder
destination_folder = 'data/final'

# if the destination folder does not exist, create it
os.makedirs(destination_folder, exist_ok=True)

# destination path (including file name change)
destination_path = os.path.join(destination_folder, 'pipeline2_final_data.csv')

# copy file and rename
shutil.copy(source_path, destination_path)