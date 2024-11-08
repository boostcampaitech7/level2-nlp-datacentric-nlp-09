import pandas as pd

"""
이 코드는 Label Noise 데이터와 증강된 데이터를 병합하는 코드입니다. 두 데이터 모두 Label에 Noise가 있기에, 병합된 데이터는 다음 Pipeline에서 Label을 복원하게 됩니다.

코드 설명:
1. `Lnoise_1200.csv`(Label Noise 데이터)와 `augmented_postprocessed.csv`(증강된 데이터) 파일을 불러옵니다.
2. 두 데이터프레임을 병합하여 하나의 데이터프레임으로 결합합니다.
3. 병합된 데이터의 총 길이를 계산하고, 이를 파일명에 반영하여 `Lnoise_Augmentation_{total_len}.csv`로 저장합니다.

"""

# load data
df = pd.read_csv('data/preprocessed/Lnoise_1200.csv')
df2 = pd.read_csv('data/augmented/augmented_postprocessed.csv')

# merge data
total_len = len(df) + len(df2)
df = pd.concat([df, df2], ignore_index=True)
df.to_csv(f'data/preprocessed/Lnoise_Augmentation_{total_len}.csv', index=False)