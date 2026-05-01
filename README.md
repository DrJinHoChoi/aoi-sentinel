# aoi-sentinel

> 자동차 전장 SMT 공정 AOI 가성불량(False Call) 자동 판별 AI 시스템

Saki AOI 검사 결과의 가성불량(약 30%)을 AI로 자동 필터링하고, 진성불량의 원인을 추론하는 파이프라인.

## 개요

| 모듈 | 입력 | 출력 | 기술 |
|------|------|------|------|
| 2D Detector | Saki top-view 이미지 | 부품 단위 불량 위치/유형 (오삽/역삽/쇼트) | YOLO / RT-DETR |
| 2D False-Call Classifier | 불량 ROI 크롭 | 진불량 / 가불량 + confidence | EfficientNet / ConvNeXt |
| 3D Analyzer | 높이맵·포인트클라우드 | 들뜸·납량 정량 + 판정 | PointNet++ / Depth CNN |
| RAG Reasoner | 판정 이력 + 현재 검사 결과 | 원인 추론 + 유사 이력 | Local LLM + Vector DB |

## 디렉토리 구조

```
aoi_sentinel/
├── data/             # Saki 결과 파서, 데이터셋, ROI 크롭
├── models/
│   ├── detector_2d/      # 부품 단위 불량 탐지
│   ├── classifier_2d/    # 가성불량 필터 (핵심)
│   ├── analyzer_3d/      # 3D 들뜸/납량 분석
│   └── rag/              # 판정 이력 RAG
├── pipeline/         # 통합 추론 파이프라인
├── train/            # 학습 스크립트
├── eval/             # 평가 (혼동행렬, 가성불량 감소율 등)
└── serve/            # FastAPI 추론 서버

configs/              # YAML 설정
scripts/              # 데이터 전처리·라벨링 보조
tests/
notebooks/            # EDA
docs/
```

## Phase

- **Phase 1 (현재)**: 2D 가성불량 필터 — ROI binary classifier 우선
- **Phase 2**: 2D Object Detection (오삽/역삽/쇼트 분류)
- **Phase 3**: 3D 들뜸/납량 분석
- **Phase 4**: LLM+RAG 원인 추론

## 설치

```bash
cd aoi-sentinel
pip install -e ".[dev]"
```

GPU 학습 시:
```bash
pip install -e ".[train]"
```

## 라이선스

Proprietary — internal use only.
