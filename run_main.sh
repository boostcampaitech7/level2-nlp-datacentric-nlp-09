#!/bin/bash

python /data/ephemeral/home/level2-nlp-datacentric-nlp-09/src/main/preprocessing/step1_data_split.py
python /data/ephemeral/home/level2-nlp-datacentric-nlp-09/src/main/preprocessing/step2_cleaning_filtering.py
python /data/ephemeral/home/level2-nlp-datacentric-nlp-09/src/main/preprocessing/step3_label_correction.py
python /data/ephemeral/home/level2-nlp-datacentric-nlp-09/src/main/preprocessing/step4_unified_label.py
# python /data/ephemeral/home/level2-nlp-datacentric-nlp-09/src/main/augmentation/step5_synthetic_data.py
python /data/ephemeral/home/level2-nlp-datacentric-nlp-09/src/main/augmentation/step6_data_cleaning.py
python /data/ephemeral/home/level2-nlp-datacentric-nlp-09/src/main/augmentation/step7_add_label.py
python /data/ephemeral/home/level2-nlp-datacentric-nlp-09/src/main/augmentation/step8_concat_csv.py
