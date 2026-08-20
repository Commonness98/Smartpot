# 🌱 스마트팜 에이전트 AI

> **온실 에너지 수요 예측 및 작업 일정 권고를 위한 에이전트 AI**
> 지능형 스마트농업 GrandICT 연구센터 · 2026 창의자율과제

실제 온실의 환경 데이터(기후·CO2·기상)와 에너지(난방+전기) 소비를 머신러닝/딥러닝으로
분석하고, 자연어 처리(NLP)와 추론·계획(Reasoning & Planning) 기능을 갖춘 에이전트 AI가
농업인에게 **방제·정비 등 비필수 작업의 최적 시점**을 자연어로 안내하는 대화형 웹
서비스입니다. 새 데이터를 업로드하면 **자동 파라미터 튜닝 에이전트**가 하이퍼파라미터를
스스로 탐색해 재학습합니다.

## 시스템 구조 (연구계획서 4계층)

```
User Interface & Q&A (Streamlit)          ← 목표④  app/
        ▲  자연어 질문 / 권고안 답변
AI Agent Core (NLP + Reasoning & Planning  ← 목표③  smartfarm/agent/
        + 자동 파라미터 튜닝)                        (core, planner, schema_mapper, tuner)
        ▲  예측 결과 + 전기요금(TOU) 취합
예측 모델 (일일 에너지수요 LSTM)            ← 목표②  smartfarm/models/
        ▲
데이터 계층 (수집·정제·적재)               ← 목표①  smartfarm/data_pipeline.py, db/
```

## 기술 스택

`Python 3.14` · `PyTorch`(LSTM) · `MariaDB`(SQLite 폴백) · `LM Studio`(로컬 LLM) ·
`Streamlit` · `Plotly` · `GitHub`

> TensorFlow는 Python 3.14 미지원이라 예측 모델은 **PyTorch**로 구현했습니다.

## 프로토타입 데이터셋

실제 라즈베리파이/Jetson 센서 수집 이전 단계로, 실제 온실에서 수집된 공개 데이터셋
**WUR(Wageningen University & Research) Autonomous Greenhouse Challenge, 2nd Edition
(2019)**을 기본 데이터셋으로 사용합니다. 네덜란드 Bleiswijk의 상업용 유리온실 6개
구획에서 6개월간(2019-12 ~ 2020-05) 실제로 토마토를 재배하며 수집한 기후(온도·습도·
CO2·광량)와 **실측 자원 소비(난방+전기, 일 단위)** 데이터입니다.
- 출처: https://data.4tu.nl/datasets/8b35675d-7549-49cc-a8c3-dedec8d3f23e
- 기본값으로는 6개 구획 중 WUR 자체 기준(대조군) 구획인 `Reference` 팀 데이터를 사용합니다.
- 실측 에너지(난방+전기)가 **일(day) 단위**로만 제공되어, 기후 데이터를 일 평균으로
  집계해 맞춥니다. 그 결과 표준 스텝 단위가 "일"이 되어(과거 10분 간격 대체 데이터와
  다름), 예측도 "다음 날 하루치 에너지 수요"를 대상으로 합니다.
- 단일 시즌(겨울→봄)만 포함하는 소규모 데이터라 학습 구간(겨울, 난방 수요 高)과
  평가 구간(봄, 난방 수요 低)의 분포 차이가 커서 일반화 성능(R²)이 낮게 나옵니다.
  다년치 데이터 확보 시 개선될 것으로 예상되는 프로토타입 단계의 한계입니다.
- `📁 데이터 학습` 탭에서 임의의 스마트팜 CSV(예: 다른 온실 구획, 자체 센서 로그)를
  올려 대체·추가할 수 있습니다.

## 빠른 시작

```bash
pip install -r requirements.txt

python scripts/00_download_data.py     # WUR 온실 데이터셋 다운로드(.7z) + 압축 해제
python scripts/01_load_data.py         # DB(sensor_readings) 적재
python -m smartfarm.agent.tuner --trials 6 --epochs 15   # 자동 파라미터 튜닝 학습
python -m streamlit run app/app.py     # 대화형 웹 UI 실행
```

위 세 단계는 `python scripts/run_all.py --epochs 15 --trials 6` 로 한 번에 실행할 수도
있습니다.

### MariaDB로 전환 (선택)

```bash
mysql -u root -p < db/schema_mariadb.sql
# .env 에 아래 추가
DB_URL=mysql+pymysql://user:password@localhost:3306/smartfarm
```

### LM Studio 연동 (선택)

LM Studio에서 로컬 LLM 서버를 켜면(기본 `http://localhost:1234/v1`) 에이전트가
자동으로 자연어 응답을 생성합니다. 꺼져 있으면 규칙 기반 응답으로 폴백합니다.
`.env`에서 `LM_STUDIO_BASE_URL`, `LM_STUDIO_MODEL` 로 오버라이드할 수 있습니다.

## 프로젝트 구조

```
smartfarm/
├── config.py              # 설정 (DB, LM Studio, TOU 요금)
├── schema.py              # 표준 스키마(역할↔컬럼 분리, canonical 특징)
├── registry.py            # 데이터셋/모델 레지스트리(다중 보관·선택)
├── db.py                  # SQLAlchemy 접근 계층
├── data_pipeline.py       # 원본 → 표준 컬럼 → DB 적재 (목표①)
├── models/                # 일일 에너지수요 예측 (목표②)
│   ├── power_lstm.py      #   LSTM 모델 + 특징/시퀀스
│   ├── train.py           #   학습 (_train_once: 저장 없는 1회 학습 / train: 저장까지)
│   └── predict.py         #   예측 + 최근 실측 시계열
└── agent/                 # AI Agent Core (목표③)
    ├── llm_client.py      #   LM Studio/Ollama 연동
    ├── schema_mapper.py   #   업로드 컬럼 자동분류(규칙+LLM 하이브리드)
    ├── tuner.py           #   자동 파라미터 튜닝(하이퍼파라미터 랜덤 서치, RMSE+R² 기준)
    ├── planner.py         #   추론·계획 (내일 수요 전망 + 작업 일정 권고)
    └── core.py            #   의도 분류 + 질의 → 권고안 오케스트레이션
app/app.py                 # Streamlit 대화형 UI + 데이터 학습 (목표④)
db/schema_mariadb.sql      # MariaDB 스키마
scripts/                   # 데이터 다운로드/적재/일괄실행/초기화 스크립트
```

## 데이터 온보딩 (임의 CSV 학습 + 자동 파라미터 튜닝)

`📁 데이터 학습` 탭에서 **아무 컬럼 구성의 CSV**를 올리면:
1. 에이전트가 각 컬럼의 역할(전력타깃/실내온도/습도/CO2/외부기상 등)을 **자동 분류**(규칙+LLM)
2. 사용자가 표에서 **확인·수정** 후 "수락하고 학습"
3. 표준 컬럼으로 변환·적재 → **자동 파라미터 튜닝 에이전트**(`smartfarm/agent/tuner.py`)가
   hidden size/layers/dropout/learning rate/seq_len 조합을 여러 개 무작위 탐색해, RMSE와
   R²를 함께 반영한 점수가 가장 좋은 조합으로 LSTM 재학습 → 데이터셋으로 **등록·전환**

물리 컬럼을 논리 역할로 매핑해 **고정 차원의 표준 특징**으로 변환하므로, 컬럼이 달라도
**코드 수정 없이** 재학습만으로 새 데이터에 적용된다. 여러 데이터셋을 등록하고 사이드바에서
전환할 수 있다. 자동 튜닝을 끄고 수동으로 epoch만 지정해 학습할 수도 있다.

## 향후 고도화 (Phase 2)

- 학습된 모델을 **라즈베리파이 / NVIDIA Jetson**에 배포하여 실제 스마트팜 적용
- 도메인 지식을 결합한 **온톨로지 기반** 의사결정 시스템으로 확장
- 기상청 API 실시간 연동 및 다변량 기상 예측 모델 추가
- WUR 챌린지의 다른 5개 구획/여러 시즌 데이터를 결합해 계절 간 분포 차이로 인한
  일반화 성능 저하 개선
- Optuna 등 베이지안 탐색 기반 자동 튜닝으로 고도화(현재는 경량 랜덤 서치)

## 성능 (프로토타입)

일일 에너지수요 예측 LSTM(자동 튜닝 채택 조합: hidden=128, layers=1, dropout=0.2,
lr=2e-3, seq_len=7일): **Test RMSE ≈ 1.7**(에너지 단위) · **R² ≈ -9.9**(1일 후 예측 기준).

R²가 낮은 이유는 위 데이터셋 설명에 적은 대로, 166일(약 5.5개월)치 단일 시즌 데이터라
학습 구간(겨울)과 평가 구간(봄)의 에너지 소비 분포가 크게 달라 일반화가 어렵기 때문이다
(학습 구간 평균 ≈5.2, 평가 구간 평균 ≈1.5). 코드/파이프라인은 정상 동작하며, 다년치
데이터로 교체하면 성능이 개선될 것으로 예상된다.
