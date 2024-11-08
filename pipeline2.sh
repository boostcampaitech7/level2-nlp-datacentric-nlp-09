#!/bin/bash

"""
이 pipeline은 ...

필요한 것은 ...
"""

# Shell script to execute the Python scripts sequentially
python src/pipeline2/\(1\)divide_Tnoise_and_Lnoise.py
python src/pipeline2/\(2\)back-translate_Tnoise.py
python src/pipeline2/\(3\)compare_Tnoise_and_BT.py
python src/pipeline2/\(4\)recover_Tnoise.py
python src/pipeline2/\(5\)augment_headlines.py
python src/pipeline2/\(6\)post-process_augmented_data.py
python src/pipeline2/\(7\)concat_Lnoise_augmented_data.py
python src/pipeline2/\(8\)recover_label_v1.py
python src/pipeline2/\(9\)recover_label_v2.py
python src/pipeline2/\(10\)save_final_data.py
