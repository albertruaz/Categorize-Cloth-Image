# Categorize-Coth-with-MLP

## 목적

임베딩 벡터를 통해, 옷의 카테고리 분류 모델 구현

conda activate newenv

nohup python train.py > nohup.out 2> nohup.err &

ps aux | grep "nohup python" | grep -v grep
