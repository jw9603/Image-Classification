#!/bin/bash

# 현재 날짜와 시간을 파일명에 추가하여 저장
dt=`date '+%Y%m%d_%H%M%S'`

# 결과 모델을 저장할 디렉토리 생성
mkdir -p ./result_model

# 매개변수 설정
bs=64                # 배치 크기
epochs=30            # 에포크 수
train_ratio=0.8      # 학습 데이터 비율

# Python 스크립트 실행
python train.py \
    --model "./result_model/mnist_classification_${dt}_bs${bs}_epoch${epochs}_ratio${train_ratio}.pth" \
    --batch_size $bs \
    --n_epochs $epochs \
    --train_ratio $train_ratio
