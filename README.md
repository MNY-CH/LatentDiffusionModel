## Latent Diffusion Models

### 논문 제목 및 저자

* **논문**: High-Resolution Image Synthesis with Latent Diffusion Models (arXiv:2112.10752)
* **저자**: Robin Rombach\*, Andreas Blattmann\*, Dominik Lorenz, Patrick Esser, Björn Ommer


<p align="center">
  <img src="assets/modelfigure.png" alt="LDM" />
</p>

---

## 설치 방법

1. **Conda 환경 구축**

   ```bash
   conda env create -f environment.yaml
   conda activate ldm
   ```

   * 위 명령어를 통해 ldm 가상환경 생성 및 activate


---

## Pretrained Models

### 1. 모델 다운로드

```bash
bash scripts/download_models.sh
```

* `models/ldm/<모델명>` 폴더에 가중치가 저장됨

### 2. 텍스트-이미지(Text-to-Image) 샘플 생성

```bash
python scripts/txt2img.py \
  --prompt "a sunset behind a mountain range, oil painting" \
  --ddim_eta 0.0 \
  --n_samples 4 \
  --n_iter 4 \
  --scale 5.0 \
  --ddim_steps 50
```

* `--prompt`: 생성할 이미지 설명
* `--n_samples`, `--n_iter`: 샘플 개수, 반복 횟수
* `--scale`: Guide scale
* `--ddim_steps`: Diffusion step 조절

### 3. 이미지 인페인팅(Inpainting)

```bash
python scripts/inpaint.py \
  --indir data/inpainting_examples/ \
  --outdir outputs/inpainting_results
```

* `indir`: 원본 이미지(`*.png`)와 마스크(`<파일명>_mask.png`) 폴더
* `outdir`: 결과 저장 폴더

### 4. Unconditional 샘플링

```bash
CUDA_VISIBLE_DEVICES=0 \
python scripts/sample_diffusion.py \
  -r models/ldm/<모델명>/model.ckpt \
  -l logs/your_log_dir \
  -n 10 \
  --batch_size 4 \
  -c 100 \
  -e 0.0
```

* `-r`: 체크포인트 경로
* `-n`: 생성할 샘플 수
* `--batch_size`: 배치 크기
* `-c`, `-e`: Diffusion step 조절 및 eta 값

---

## 자체 LDM 학습 (Train your own LDMs)

### 1. 데이터 준비

* **CelebA-HQ, FFHQ**: `taming-transformers`의 지침에 따라 다운로드 및 전처리
* **LSUN**: 다음 링크에서 데이터셋을 다운로드. (https://github.com/fyu/lsun)
* **ImageNet**: 다음 링크에서 데이터셋을 다운로드. (https://image-net.org/download-images.php)

### 2. Autoencoder 사전 학습

```bash
CUDA_VISIBLE_DEVICES=0 \
python main.py \
  --base configs/autoencoder/autoencoder_kl_8x8x64.yaml \
  -t \
  --gpus 0,
```

* `configs/autoencoder` 내부에서 YAML 파일 선택 후 실행

### 3. LDM 모델 학습

```bash
CUDA_VISIBLE_DEVICES=0 \
python main.py \
  --base configs/latent-diffusion/celebahq-ldm-vq-4.yaml \
  -t \
  --gpus 0,
```

* `configs/latent-diffusion` 내부에서 원하는 데이터셋 설정 YAML 파일 선택
* 학습 로그와 체크포인트는 `logs/<날짜>_<설정명>/`에 저장

### 4. 학습 결과 확인 및 샘플링

* `logs` 폴더에서 학습 그래프 및 샘플 이미지 확인
* 최종 `.ckpt` 파일을 바탕으로 추론

---
