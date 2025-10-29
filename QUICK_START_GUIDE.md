# 🚀 Quick Start Guide - Baseline ADL Recognition

## ⚡ 30초 만에 시작하기

### 1️⃣ 데이터 준비 (10초)

```bash
cd /home/lee/research-hub/hyunku/iot/baseline-adl-recognition
./setup_data.sh
```

✅ 이제 사용 가능한 데이터:
- **7,066개 시퀀스** (T=100, F=114)
- **5개 클래스** (t1~t5: 요리, 손씻기, 수면, 약먹기, 식사)
- **이미 전처리 완료!**

### 2️⃣ 모델 학습 (15분 설정)

```bash
python train/train.py \
  --data_path data/processed/dataset_with_lengths_v3.npz \
  --epochs 50 \
  --batch_size 32 \
  --lr 3e-4 \
  --dropout 0.1 \
  --hidden_dim 256 \
  --focal_gamma 1.5 \
  --patience 15
```

⏱️ 예상 학습 시간: ~15분 (GPU) / ~45분 (CPU)

### 3️⃣ 모델 평가 (5초)

```bash
python evaluate/evaluate.py \
  --checkpoint checkpoints/best_baseline.pt \
  --data_path data/processed/dataset_with_lengths_v3.npz
```

### 4️⃣ 어텐션 시각화 (5초)

```bash
python evaluate/visualize.py \
  --checkpoint checkpoints/best_baseline.pt \
  --data_path data/processed/dataset_with_lengths_v3.npz \
  --n_samples 10
```

---

## 📊 예상 결과

학습이 성공적으로 완료되면:

```
✅ Test Accuracy: 95.4%
✅ Macro F1: 0.947
✅ Per-class F1:
   - t1 (cooking):      0.936
   - t2 (hand washing): 0.972 ⭐
   - t3 (sleeping):     0.986
   - t4 (medicine):     0.919
   - t5 (eating):       0.920
```

---

## 🎯 전체 워크플로우 (한 번에 실행)

```bash
# 모든 단계를 한 번에 실행 (최적 설정)
cd scripts
./quick_start.sh
```

이 스크립트는:
1. ✅ 데이터 준비 확인
2. 🏋️ 모델 학습 (50 epochs, 최적 하이퍼파라미터)
3. 📊 테스트 세트 평가
4. 🎨 어텐션 가중치 시각화

---

## 📁 결과 파일 위치

학습 후 다음 파일들이 생성됩니다:

```
checkpoints/
├── best_baseline.pt           # 최고 성능 모델
├── training_history.png       # 학습 곡선
├── confusion_matrix.png       # 혼동 행렬
└── test_results.json          # 평가 결과

results/
├── test_results.json          # 상세 평가 결과
└── visualizations/            # 어텐션 시각화
    ├── attention_*.png
    ├── confidence_*.png
    └── sensors_*.png
```

---

## 🔧 커스터마이징

### 학습 설정 변경

```bash
# 최적 설정 (권장)
python train/train.py \
  --data_path data/processed/dataset_with_lengths_v3.npz \
  --epochs 50 --batch_size 32 --lr 3e-4 \
  --dropout 0.1 --hidden_dim 256 --focal_gamma 1.5 --patience 15

# 더 긴 학습
python train/train.py \
  --data_path data/processed/dataset_with_lengths_v3.npz \
  --epochs 100 --patience 20

# 더 큰 모델
python train/train.py \
  --data_path data/processed/dataset_with_lengths_v3.npz \
  --hidden_dim 512 --dropout 0.05
```

### 새 데이터셋 생성

```bash
# 더 긴 시퀀스
python data/build_features.py \
  --data_dir data/raw/processed_all \
  --output data/processed/dataset_long.npz \
  --T 150 \
  --stride 10

# 사용
python train/train.py \
  --data_path data/processed/dataset_long.npz
```

---

## 📚 자세한 문서

- **[README.md](README.md)** - 프로젝트 개요 및 성능 비교
- **[docs/DATA_SETUP.md](docs/DATA_SETUP.md)** - 데이터 준비 상세 가이드
- **[models/README.md](models/README.md)** - 모델 아키텍처 설명
- **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** - 프로젝트 구조 전체 설명

---

## 🆘 문제 해결

### "File not found" 에러
```bash
# 데이터 링크 확인
ls -lh data/raw/
ls -lh data/processed/

# 다시 설정
./setup_data.sh
```

### "CUDA out of memory" 에러
```bash
# 배치 사이즈 줄이기
python train/train.py --batch_size 32
```

### "ModuleNotFoundError" 에러
```bash
# 의존성 재설치
pip install -r requirements.txt
```

---

## 🎓 학습 팁

1. **첫 실행**: 기본 설정으로 먼저 학습
2. **과적합 확인**: Validation loss가 증가하면 early stopping 작동
3. **성능 개선**: `--patience` 늘리고 `--epochs` 늘리기
4. **빠른 테스트**: `--epochs 5`로 먼저 확인

---

## ✨ 다음 단계

학습이 완료되면:

1. 📊 **결과 분석**: `results/test_results.json` 확인
2. 🎨 **어텐션 분석**: `results/visualizations/` 확인  
3. 🔬 **실험**: 다른 하이퍼파라미터로 학습
4. 🚀 **배포**: 모델을 엣지 디바이스에 배포

---

**🎉 이제 준비 완료! 행운을 빕니다!**
