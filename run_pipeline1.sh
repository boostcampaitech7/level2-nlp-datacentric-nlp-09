#!/bin/bash

# Python 파일들이 있는 경로
PYTHON_SCRIPTS_PATH="./src/pipeline1"

# 실행할 Python 파일 이름들 (확장자 포함)
PYTHON_FILES=(
    "preprocessing/replace_3dot_with_space.py"
    "preprocessing/filter_by_asciiCondition.py"
    "preprocessing/filter_by_asciiRatio.py"
    "preprocessing/preprocess_ascii_engWord.py"
    "preprocessing/extract_unique_ids.py"
    "preprocessing/self_training.py"
    "preprocessing/combine_row.py"
    "preprocessing/preprocess_morpheme.py"
)

# 각 Python 파일 실행
for file in "${PYTHON_FILES[@]}"
do
    echo "Running $file..."
    python "$PYTHON_SCRIPTS_PATH/$file"
    if [ $? -ne 0 ]; then
        echo "Error occurred while running $file. Stopping execution."
        exit 1
    fi
done

echo "All scripts executed successfully."
