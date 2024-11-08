#!/bin/bash

"""

- 대회 개요

대회에서 사용되는 데이터셋은 KLUE 공식 사이트에서 제공하는 KLUE-TC(YNAT) 데이터셋과 같은 포맷을 가집니다. 제공되는 총 학습 데이터는 2,800개이며, 테스트 데이터는 30,000개 입니다.
기존 KLUE-YNAT 데이터셋 중 일부를 학습데이터로, 일부를 테스트데이터로 사용합니다.
대부분의 학습데이터에는 noise (text 컬럼 값에 노이즈 및 라벨 변경)가 섞인 데이터가 일부 섞여있습니다.
노이즈가 심한 데이터를 필터링하고, 노이즈가 심하지 않은 데이터를 복구하고, 잘못된 라벨이 있는 데이터를 찾아서 고치거나 필터링하는 것이 대회의 핵심입니다.


Pipeline2의 shell script는 주어진 raw train 데이터를 레이블 노이즈(Label Noise)와 텍스트 노이즈(Text Noise)가 포함된 두 개의 파일로 분할하는 코드입니다.
각각의 노이즈는 앞으로 Pipeline에서 독립적으로 처리되어 최종적으로 레이블 노이즈와 텍스트 노이즈가 복구되고, 추가로 데이터가 증강되어 최종적인 학습 데이터셋이 생성됩니다.


- Pipeline2 처리 과정

텍스트 노이즈 복구
1. divide_Tnoise_and_Lnoise.py: 레이블 노이즈와 텍스트 노이즈가 포함된 데이터를 분리합니다.
2. back-translate_Tnoise.py: 텍스트 노이즈를 Back-Translation을 통해 복구합니다. (DeepL API Key 필요)
3. compare_Tnoise_and_BT.py: 텍스트 노이즈와 Back-Translation을 통해 복구된 데이터를 비교하여 더 나은 데이터를 선택합니다.
4. recover_Tnoise.py: 선택된 데이터를 Generation Model을 통해 텍스트 노이즈를 제거합니다.

데이터 증강
5. augment_headlines.py: Generation Model을 통해 새로운 텍스트 데이터를 증강합니다.
6. post-process_augmented_data.py: 증강된 데이터를 후처리합니다.

레이블 노이즈 복구
7. concat_Lnoise_augmented_data.py: 증강된 데이터와 레이블 노이즈가 포함된 데이터의 레이블을 새로 맞추기 위해 증강된 데이터를 합칩니다.
8. recover_label_v1.py: PLM을 (텍스트 노이즈 복구 데이터셋)으로 fine-tuning하고 레이블 노이즈를 추론합니다.
9. recover_label_v2.py: PLM을 (텍스트 노이즈 복구 데이터셋 + 레이블 노이즈 복구 데이터셋)으로 새로 fine-tuning하고 레이블 노이즈를 재추론합니다.

최종 데이터셋 저장
10. save_final_data.py: 최종적으로 모든 노이즈가 복구된 데이터셋을 저장합니다.
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
