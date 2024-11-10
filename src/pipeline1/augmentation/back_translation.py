import pandas as pd
import deepl
from tqdm import tqdm

# DeepL API 인증키 설정
auth_key = "key"  # 본인의 DeepL API 인증키로 변경하세요
translator_deepl = deepl.Translator(auth_key)

# DeepL을 사용한 역번역 함수 정의
def back_translate_deepl(text):
    try:
        # Step 1: 한국어 -> 영어 번역
        en_result = translator_deepl.translate_text(text, target_lang="EN-US")
        if not en_result.text:
            return None  # 번역 실패 시 None 반환
        
        # Step 2: 영어 -> 한국어 역번역
        ko_result = translator_deepl.translate_text(en_result.text, target_lang="KO")
        return ko_result.text  # 성공 시 역번역 결과 반환
    
    except Exception as e:
        print(f"Error in back-translation for text '{text}': {e}")
        return None  # 에러 발생 시 None 반환

# CSV 파일을 받아 역번역하여 데이터 증강 수행 함수 정의
def augment_csv_with_back_translation(input_csv_path, output_csv_path):
    # CSV 파일 로드
    df = pd.read_csv(input_csv_path)
    
    # tqdm을 사용하여 역번역 진행 상태 표시
    tqdm.pandas()
    
    # 역번역 수행
    df['augmented_text'] = df['text'].progress_apply(back_translate_deepl)
    
    # 역번역된 결과 중 None 값을 제거하고 원본 데이터와 결합
    augmented_df = df.dropna(subset=['augmented_text']).copy()
    augmented_df['text'] = augmented_df['augmented_text']  # 역번역 결과를 기존 열 이름으로 맞춤
    augmented_df = augmented_df.drop(columns=['augmented_text'])  # 보조 열 삭제
    
    # 원본 데이터와 역번역 데이터 결합
    final_df = pd.concat([df.drop(columns=['augmented_text']), augmented_df], ignore_index=True)
    
    # 결과를 CSV로 저장
    final_df.to_csv(output_csv_path, index=False)
    print(f"Augmented data saved to {output_csv_path}")

# 예시 파일 경로
input_csv_path = "./data/preprocessed/unique_ids_engWord_output.csv"       # 입력 CSV 파일 경로
output_csv_path = "./data/preprocessed/unique_ids_engWord_output_backtransAug.csv"  # 출력 CSV 파일 경로
