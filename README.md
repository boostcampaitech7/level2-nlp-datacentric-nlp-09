# Topic Classification Project: A Data-Centric Approach for Classifying Text Topics Without Modifying Model Architecture

The KLUE-Topic Classification benchmark focuses on classifying the topic of a news article based on its headline. Each natural language data point is labeled with one of several topics, including Society, Sports, World, Politics, Economy, IT/Science, and Society.

## Competition Rules

In alignment with the data-centric approach, participants must improve model performance solely through data modification without altering the baseline model code. Use of paid API services is prohibited. Any modifications to the baseline code will result in disqualification from the leaderboard post-competition review.

Wrap-up Reports : [ODQA with RAG](https://drive.google.com/drive/u/0/folders/19lW_Dohoj2oBXjuhOV6JOjhfx69MCznU)
<br>
<br>

## Overview

### 1. Environment Setup

We have four different methods for preprocessing and augmenting the original `data/raw/train.csv`, stored in `src/main`, `src/pipeline1`, `src/pipeline2`, and `src/pipeline3`. First, navigate to the respective directory and set up the environment by running the `requirements.txt` or `environment.yml` file. Then, execute the pipelines using the shell scripts named accordingly.

<br>
System requirements:
Ubuntu-20.04.6 LTS

Each branch (`main`, `pipeline1`, `pipeline2`, `pipeline3`) specifies the exact Python and PyTorch versions used for its respective model.

<br>

### 2. Data

![스크린샷 2024-10-28 오후 5.18.05.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/b1dce0c2-a152-44e2-bf4a-278dfea60250/72aa2ed2-e73e-493b-8cc3-807dab22a56e/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-10-28_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_5.18.05.png)

- **Data attribute :** The random noise and labeling noise are mutually exclusive.
- **Test Dataset :** consists of 30,000 data points, each containing an ID and Text.

<br>

### 3. Pipeline Stages

- main
    
    Branch: `main`
    
    ## Preprocessing
    
    ### Data Splitting
    
    - Split the training data into "noise-free" and "noisy" datasets based on the presence of noise in the text. After examining the raw data, noise was found to consist of replacing Korean text and spaces with English letters, numbers, and special characters (ASCII codes 33–126). Texts containing over 20% of such characters were labeled as "noisy."
        - Text Noise Data: 1,608 samples
        - Label Error Data: 1,192 samples
    
    ### Random Noise Processing
    
    - For texts with noise, extracted only nouns using the Mecab morphological analyzer, removing samples without nouns. (Text Noise Data reduced from 1,608 to 1,606 samples)
    - Similarly, for label error data, extracted nouns from noise-free text to prepare for relabeling.
    
    ### Labeling Error Processing
    
    - Trained a `klue/base-bert` model on preprocessed text noise data. Applied this model to relabel noise-free noun-extracted data.
    - For consistency, relabeled texts were mapped back to their original IDs, updating the clean dataset's labels accordingly.
    
    ## Augmentation
    
    ### Synthetic Data
    
    - Used a large language model (`allganize/Llama-3-Alpha-Ko-8B-Instruct`) to generate additional news headlines across various topics for augmentation.
        - Noise Processing: Verified that generated texts also contained noise; extracted nouns using Mecab (Final Augmentation Data: 7,338 samples).
        - Labeling: Labeled synthetic data in the same manner as above, aligning with relabeling methodology.

<br>

### 4. Evaluation Metrics

To assess model performance, the following metrics are used:

- **Accuracy**
- **macro** F1 Score:  F1 Score gives a partial score by considering word overlap between the prediction and the true answer.

<br>

### 5. Results

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/b1dce0c2-a152-44e2-bf4a-278dfea60250/0752e147-b102-498f-bb2a-c0f6e15be5e2/image.png)

- Accuracy: 84.20%
- macro F1 Score: 84.05%

*These metrics provide insight into both the accuracy and partial correctness of the model’s predictions across all stages of the pipeline.*
