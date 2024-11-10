import pandas as pd
import numpy as np
import os



def switch_text(original_text):
    mid_index = len(original_text) // 2  
    
    while mid_index < len(original_text) and original_text[mid_index] != ' ':
        mid_index += 1
        
    switched_text = original_text[mid_index:] + ", " + original_text[:mid_index]
    
    return switched_text




BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, 'data/preprocessed')
filtered_df = pd.read_csv(os.path.join(DATA_DIR, 'text_fix_nouns.csv'))


filtered_df['switched_text'] = filtered_df['text'].apply(switch_text)


new_rows = pd.DataFrame({
    'ID': filtered_df['ID'],  
    'text': filtered_df['switched_text'],      
    'target': filtered_df['target']               
})

df_combined = pd.concat([filtered_df, new_rows], ignore_index=True)

filtered_df.to_csv("data/final/pipeline2_final.csv", index=False, encoding='utf-8-sig')