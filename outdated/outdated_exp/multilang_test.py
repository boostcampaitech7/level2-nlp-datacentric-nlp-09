from transformers import pipeline
import pandas as pd

# Load a multi-class classification pipeline - if the model runs on CPU, comment out "device"
classifier = pipeline("text-classification", model="classla/multilingual-IPTC-news-topic-classifier", device=0, max_length=512, truncation=True)

df = pd.read_csv('data/preprocessed/mask2goldentruth.csv')
texts = df['text'].tolist()

# # Example texts to classify
# texts = [
#     """갤노트8 주말 27만대 개통…시장은 불법 보조금 얼룩""",
#     """美성인 6명 중 1명꼴 배우자·연인 빚 떠안은 적 있다""",
#     """아시안게임 목소리 높인 박항서 베트남이 일본 못 이길...""",
#     """유엔 리비아 내전 악화·국제적 확산 우려…아랍연맹 긴급회의종합""",
#     """공룡 파충류보다는 타조·매 등 새에 가깝다""",
#     """네이버 모바일 연예판에도 AI 콘텐츠 추천 시스템 적용""",
#     """코스피 미국발 악재에 또 급락…2360대로 털썩""",
#     """러 우주로켓 벼락 맞고도 멀쩡…탑재 위성 정상궤도 올려""",
#     """KB증권 농심 4분기 라면 부문 실적개선…목표주가↑""",
#     """그래픽 내년 코스피 전망치""",
#     """위안부 피해자 그림 선생님이 전하는 이야기""",]

# Classify the texts
results = classifier(texts)

# Output the results
# print()
# for text, result in zip(texts, results):
#     print(text)
#     print(result)
#     print()

## Output
## {'label': 'sport', 'score': 0.9985264539718628}
## {'label': 'disaster, accident and emergency incident', 'score': 0.9957459568977356}

english_to_korean = {
    'disaster, accident and emergency incident': 3,
    'human interest': 0,
    'politics': 2,
    'education': 3,
    'crime, law and justice': 3,
    'economy, business and finance': 5,
    'conflict, war and peace': 6,
    'arts, culture, entertainment and media': 0,
    'labour': 5,
    'weather': 0,
    'religion': 3,
    'society': 3,
    'health': 3,
    'environment': 3,
    'lifestyle and leisure': 0,
    'science and technology': 4,
    'sport': 1
}

df['target'] = [english_to_korean[result['label']] for result in results]

df.to_csv('data/preprocessed/GT_label_restored.csv', index=False) # processed된 파일 이름 폴더에 결과 저장