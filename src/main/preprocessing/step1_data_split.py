import pandas as pd

# CSV 파일 불러오기
df = pd.read_csv('data/raw/train.csv')

# 영어 + 숫자 + 특수문자 비율을 계산하는 함수 (공백 미포함)
def non_korean_ratio(text):
    non_korean_count = sum(1 for char in text if (33 <= ord(char) <= 126))
    return non_korean_count / len(text) if len(text) > 0 else 0

# 영어 + 숫자 + 특수문자 비율이 20% 미만인 데이터와 그 외 데이터를 각각 필터링
df_under_10 = df[df['text'].apply(non_korean_ratio) < 0.2]
df_over_10 = df[df['text'].apply(non_korean_ratio) >= 0.2]

# 결과를 각각 새로운 CSV 파일로 저장
df_under_10.to_csv('data/preprocessed/train_clean_v1.csv', index=False)
df_over_10.to_csv('data/preprocessed/train_noise_v1.csv', index=False)

##################################################################

clean = "data/preprocessed/train_clean_v1.csv"
noise = "data/preprocessed/train_noise_v1.csv"

df_clean = pd.read_csv(clean)
df_noise = pd.read_csv(noise)

# 'ID' 열에서 특정 패턴으로 끝나는 데이터를 필터링
df_noise_new = df_clean[df_clean['ID'].astype(str).str.endswith(
    ('00030', '00090', '00114', '00139', '00188', '00203', '00257', '00258', '00334', '00453', '00543', '00606', '00686',
     '00774', '00775', '00778', '00782', '00801', '00815', '00818', '00824', '00895', '00906', '00918', '00920', '00932',
     '00945', '01161', '01193', '01226', '01307', '01374', '01483', '01568', '01668', '01669', '01682', '01790', '01881',
     '01889', '01904', '01918', '02096', '02298', '02322', '02358', '02424', '02453', '02482', '02592', '02663', '02693',
     '02742', '02756', '02781')
    )]
df_clean_new = df_noise[df_noise['ID'].astype(str).str.endswith(
    ('00007', '00015', '00020', '00039', '00066', '00068', '00115', '00134', '00191', '00198', '00199', '00226', '00277',
     '00338', '00363', '00346', '00467', '00491', '00573', '00577', '00582', '00602', '00638', '00691', '00741', '00745',
     '00748', '00769', '00831', '00853', '00858', '00927', '00934', '00957', '00982', '00985', '01080', '01092', '01132',
     '01133', '01168', '01178', '01187', '01225', '01262', '01382', '01402', '01444', '01575', '01577', '01582', '01626',
     '01639', '01649', '01688', '01692', '01707', '01729', '01776', '01826', '01828', '01855', '01873', '01874', '01911',
     '01913', '01927', '02111', '02139', '02175', '02193', '02223', '02224', '02228', '02231', '02248', '02289', '02299',
     '02302', '02370', '02369', '02421', '02427', '02434', '02456', '02465', '02511', '02513', '02586', '02599', '02612',
     '02672', '02746', '02747', '02790')
    )]

# 필터링된 데이터를 각각의 파일에 추가
df_noise = pd.concat([df_noise, df_noise_new], ignore_index=True)
df_clean = pd.concat([df_clean, df_clean_new], ignore_index=True)

# 각각의 파일에서 필터링된 데이터를 제거
df_clean = df_clean[~df_clean['ID'].isin(df_noise_new['ID'])]
df_noise = df_noise[~df_noise['ID'].isin(df_clean_new['ID'])]

df_clean.to_csv('data/preprocessed/train_clean_v2.csv', index=False)
df_noise.to_csv('data/preprocessed/train_noise_v2.csv', index=False)
