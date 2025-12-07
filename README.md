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

폴더별 공통 코드

env_*.py : Grid 환경 정의

train_*.py : 학습 코드

test_*.py : 평가 코드

*_visual.py : 시각화 / 경로 출력

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
