import pandas as pd
import deepl
from googletrans import Translator
from tqdm import tqdm
import time
tqdm.pandas()

"""
이 코드는 주어진 데이터셋에 대해 Back Translation을 사용하여 데이터 전처리를 수행하는 코드입니다.
Text Noise가 포함된 데이터셋을 Back Translation을 통해 한-영-한 번역을 수행하여 노이즈를 제거합니다.

코드 설명:
1. `deepl`과 `googletrans` 라이브러리를 사용하여 한-영-한 번역을 통해 텍스트를 변환하는 두 가지 함수가 정의되어 있습니다.
   - `back_translate_deepl`: DeepL API를 이용해 텍스트를 영어로 번역한 후 다시 한국어로 번역.
   - `back_translate_google`: Google Translator를 이용해 텍스트를 영어로 번역 후 다시 한국어로 번역.
2. DeepL API를 이용한 번역 함수를 사용해 `Tnoise_1073.csv` 데이터의 텍스트 컬럼을 변환.
3. 변환된 데이터를 새로운 CSV 파일 `Tnoise_BT_1073.csv`에 저장.

주의: API 키는 실제 API 키를 사용하여야 합니다.
"""

# function to back translate using DeepL
def back_translate_deepl(text):
    result = translator_deepl.translate_text(text, target_lang="EN-US")
    if not result.text:
        return None
    result = translator_deepl.translate_text(result.text, target_lang="KO")
    return result.text

# function to back translate using Google Translator
def back_translate_google(text, src="ko", dest="en"):
    while True:
        try:
            en_result = translator_google.translate(text, dest=dest)
            ko_result = translator_google.translate(en_result.text, dest=src)
            result = ko_result.text
            return result
        except Exception as e:
            time.sleep(1)
            continue


# get auth key
auth_key = 'api key 사용' 

# select translator
translator_deepl = deepl.Translator(auth_key)
# translator_google = Translator()

# load data
noise_Tnoise_HighRatioCut = pd.read_csv('data/preprocessed/Tnoise_1073.csv')

# back translate
noise_Tnoise_HighRatioCut['text'] = noise_Tnoise_HighRatioCut['text'].progress_apply(back_translate_deepl)
# df['google'] = df['text'].progress_apply(back_translate_google)

# save
noise_Tnoise_HighRatioCut.to_csv('data/preprocessed/Tnoise_BT_1073.csv', index=False)