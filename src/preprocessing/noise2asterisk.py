import pandas as pd
import re
import argparse

argparser = argparse.ArgumentParser(description='Preprocess data')
argparser.add_argument('--input', type=str, default='data/raw/train.csv', help='input file path')
argparser.add_argument('--output', type=str, default='noise2asterisk.csv', help='output file path')

args = argparser.parse_args()


df = pd.read_csv(args.input)

# 한글, 공백을 제외한 모든 문자를 *로 대체
# 노이즈가 아닌 숫자, 영문자도 대체됨
sub = '*'
df['text'] = df['text'].apply(lambda x: re.sub(r'[^가-힣\s]', sub, x))

# 저장
df.to_csv(f'data/preprocessed/{args.output}', index=False)