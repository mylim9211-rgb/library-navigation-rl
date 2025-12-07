📚 Reinforcement Learning for Library Bookshelf Navigation
Simple → Curriculum → Advanced 단계별 난이도 확장 기반 탐색 정책 학습

**Overview**

본 프로젝트는 도서관 서가 환경에서 로봇이 목표 위치(A~G)를 스스로 탐색하는 능력을 강화학습(Deep Q-Network, Double/Dueling DQN) 기반으로 학습·평가하는 연구 프로젝트입니다.

환경의 난이도를
Simple → Curriculum → Advanced
순으로 확장하며, 정책의 일반화 성능, 안정성, seed 의존성, 타깃별 성공률 등을 체계적으로 분석하였습니다.

**프로젝트 구조(Project Structure)**
library-navigation-rl/
│
├── ★Library_Simple/         # Simple grid 환경 + 학습/테스트 코드
├── ★Library_Curriculum/     # Curriculum 학습 환경(S→Random)
├── ★Library_Advanced/       # 최종 Advanced 환경 + Double/Dueling 모델
│
├── README.md                # 이 문서
└── .gitignore


각 폴더는 다음 구성으로 이루어져 있습니다:

폴더별 코드 설명 
 
★ Library_Simple

가장 기본적인 서가 A~C로 구성된 단순 환경.
충돌 규칙 최소화, 가장 쉬운 난이도.

주요 파일 설명
Library_Simple_env.py : Simple Grid 환경 정의 장애물 없음, 구조 단순 가장 안정적으로 학습이 이루어지는 단계

library_shelf_random_start_double.pt : Simple 환경에서 Random Start 포함한 Double DQN 학습 모델 가중치

library_shelf_random_start_curriculum.pt : Curriculum로 넘어가기 직전 Robust 모델

Library_train_random_start.py : Simple 환경에서 Random Start 포함 학습 Curriculum 초기값으로 활용

Library_simple_test_summary.py : Simple 단계 전체 성공률·평균스텝·wall-hit 등 요약

Library_test_random_start.py : Random Start에서 정책 안정성 테스트 Curriculum 환경으로 넘어가기 전 정책의 generalization 검증

Library_test_simple_image.py : Sample Path(훈련 결과)를 PNG 이미지로 시각화

simple_eval_visual.py : Simple 환경의 초기 성능 검증 및 Path 비주얼라이제이션


⭐ 2. ★Library_Curriculum

중간 난이도 환경.
서가 구조는 유지하되 장애물·벤치가 추가되고 Random Start 비율이 조절되는 환경.

주요 파일 설명

env_curriculum.py : Curriculum Grid 환경 정의 Simple 대비 구조 약간 복잡, 장애물/벤치 포함 Random Start 비율을 ↑ 조절하는 Curriculum 전략 반영

library_curriculum_base.pt : Curriculum 단계에서 학습된 Base 모델 가중치

library_simple_robust.pt / library_simple_robust.pt : Simple 단계에서 수렴된 정책을 Curriculum으로 전이할 때 사용되는 Robust 초기 모델

train_curriculum_base.py : Curriculum 환경에서의 기본 학습 스크립트 Random Start 비율 0% → 80%로 증가시키며 Training 진행

train_simple_robust.py : Simple 단계 Robust 모델을 Curriculum 단계로 전이하여 학습하는 스크립트

test_simple_robust_summary.py : Curriculum 환경에서의 평균 성공률/스텝 등을 한 번에 정리하는 모듈

curriculum_eval_visual.py : Curriculum 환경 이동 경로 시각화 보고서용 Path 이미지 생성

test_simple_robust_live.py : 학습된 Curriculum 모델을 실시간으로 1회 실행하여 행동 패턴을 확인하는 파일

⭐ 3. ★Library_Advanced

가장 복잡한 서가 지형(장애물·복도·다중 타겟)에서 학습·평가하는 최종 환경.

주요 파일 설명

env_advanced.py : Advanced Grid 환경 정의 파일 장애물 배치, 서가 구조, 이동 규칙 등을 포함 Simple/Curri 대비 가장 복잡한 지형 로직 포함

library_AG_double_dueling.pt : 최종 학습된 “Double DQN + Dueling Network” 정책 모델 가중치 파일 

Library_env_train.py : Advanced 환경 학습 스크립트 Double DQN + Dueling 구조로 4,000 episode 학습 수행

Library_advanced_eval.py : 대표 Seed 기반 성능 평가 S-start / Random-start 성능 타겟별 성공률(A~G) 출력

Library_advanced_seed_eval.py : 여러 seed(예: 1, 42, 2025) 기반 Random-start 성능 측정 95% CI 계산용 스크립트

Library_test_advanced_summary.py : Advanced Grid 환경에서 전체 성공률, 평균 스텝, wall-hit, stuck-rate 정리 보고서용 핵심 지표 산출 

Library_env_test_advanced_image.py : 고정 시드로 Sample Path(로봇 이동 경로)를 시각화하여 PNG로 출력하는 파일

**학습 전략(Learning Strategy)**

1) Simple → Curriculum → Advanced 단계적 확장
단계	특징
Simple	구조 단순 / 충돌 규칙 최소화 / 기본 이동 패턴 학습
Curriculum	장애물 + Random Start 비율 조절로 일반화 강화
Advanced	복잡한 서가(A~G) + 벽/벤치 + 실제 환경과 유사한 난이도

2) Curriculum Learning 기법 적용

학습 초반: Random Start 100%

중반: Random 비율 감소 → S-start 집중 학습

후반: Generalization을 위해 Random Start 다시 증가
(0% → 80%)

3) Reward 설계

기본 이동: -0.01 ~ -0.05

벽 충돌: -1.0

목표 도달: +10 ~ +20

지름길 탐색 유도 + 불필요한 충돌 억제

**실행 방법 (How to Run)**
1) 환경 설치
pip install -r requirements.txt

2) Simple 환경 학습
python ★Library_Simple/train_simple.py

3) Curriculum 학습
python ★Library_Curriculum/train_curriculum.py

4) Advanced 학습
python ★Library_Advanced/train_advanced.py

5) 평가 코드 실행
python ★Library_Advanced/advanced_eval.py

**주요 결과(Results)**
Seed 고정 vs Random 비교

S-start 성공률: 약 83~88%

Random-start 성공률: 약 69~75%

95% CI: 69.8% ± 5.0%

→ seed 변화에도 정책이 안정적으로 일반화된 행동 패턴을 학습했음을 의미.

타깃(A~G)별 성공률

Advanced 환경 기준:

Target	성공률
A	70%
B	66%
C	70%
D	66%
E	78%
F	90%
G	46%

→ G 서가는 실제로도 구조적으로 난이도가 높음 → 연구적으로도 흥미로운 포인트.

Sample Path Visualization

<img width="899" height="349" alt="image" src="https://github.com/user-attachments/assets/56a8e1e9-74c8-4d97-b192-507b209afb34" />


results/
└── sample_path.png

**실험 재현성 (Reproducibility)**

동일 seed 설정 가능

Advanced 환경은 seed 10개 기반 CI 계산 코드 포함

모든 학습/평가 스크립트 재현 가능하게 정리됨

**향후 개선 방향(Future Work)**

Multi-agent 탐색

Actor-Critic 계열(PPO, A3C)로 성능 비교

더 현실적인 Library Map 적용

LLM 기반 reward shaping 적용 가능성 탐색

만든이

A71051 임재윤 (서강대학교 AI SW 대학원)
