
### *Simple → Curriculum → Advanced 단계별 난이도 확장 기반 강화학습 탐색 프로젝트*

본 프로젝트는 **도서관 서가(Grid) 환경에서 로봇이 목표 위치(A~G)를 탐색하는 정책을   
Deep Q-Network(DQN), Double DQN, Dueling Network 기반으로 학습**하는 연구입니다.

환경 난이도를  
**Simple → Curriculum → Advanced**  
순으로 확장하면서 학습 안정성·일반화 성능·seed 의존성 등을 체계적으로 분석했습니다.

---

# 🗂️ Project Structure

library-navigation-rl/
│
├── ★Library_Simple/
├── ★Library_Curriculum/
├── ★Library_Advanced/
│
├── README.md
└── .gitignore

각 폴더는 다음 구성으로 이루어져 있습니다:

폴더별 코드 설명 
 

각 폴더는 **환경 정의(env)**, **학습(train)**, **평가(test)**, **시각화(visual)** 파일로 구성되며  
난이도에 따라 기능이 확장됩니다.

---

# ★Library_Simple  
> **가장 기본적인 서가(A~C) + 단순 이동 규칙 기반 환경**

### 주요 파일 설명
| 파일명 | 설명 |
|-------|------|
| **Library_Simple_env.py** | Simple Grid 환경 정의 (장애물 없음, 구조 단순) |
| **Library_train_random_start.py** | Simple 환경 학습(Random Start 포함) |
| **Library_test_random_start.py** | Random-start 정책 안정성 평가 |
| **Library_test_simple_image.py** | 고정 seed 기반 Sample Path 이미지 생성 |
| **Library_simple_test_summary.py** | Simple 전체 통계(성공률/스텝/충돌률) 정리 |
| **simple_eval_visual.py** | Simple 시각화/정성적 성능 검증 |
| **library_shelf_random_start_curriculum.pt** | Curriculum 학습 초기 사용 Robust 모델 |
| **library_shelf_random_start_double.pt** | Simple Double DQN 모델 가중치 |

---

# ★Library_Curriculum  
> **장애물/벤치 추가 + Random Start 비율 조정되는 중간 난이도 환경**  
> *Simple 모델을 이어받아 일반화 능력을 강화하는 단계*

### 주요 파일 설명
| 파일명 | 설명 |
|-------|------|
| **env_curriculum.py** | Curriculum Grid 환경 정의 |
| **train_curriculum_base.py** | Curriculum 학습 메인(Random Start 0%→80%) |
| **train_simple_robust.py** | Simple Robust → Curriculum 전이 학습 |
| **curriculum_eval_visual.py** | Curriculum 환경 경로 시각화 |
| **test_simple_robust_summary.py** | Curriculum 전체 성능 요약 |
| **test_simple_robust_live.py** | 실시간 1회 에피소드 테스트 |
| **library_simple_robust.pt** | Simple 단계 Robust 초기 모델 |
| **library_curriculum_base.pt** | Curriculum 학습된 Base 모델 |

---

# ★Library_Advanced  
> **가장 복잡한 서가(A~G 전체) + 장애물 + 복도 구조 포함한 최종 난이도 환경**  
> 실제 도서관 지형에 가까운 형태로 정책의 일반화 성능 확인

### 📌 주요 파일 설명
| 파일명 | 설명 |
|-------|------|
| **env_advanced.py** | Advanced Grid 환경 정의 |
| **library_env_train.py** | Advanced 환경 학습(Double + Dueling, 4,000 episodes) |
| **Library_advanced_eval.py** | 대표 Seed 기반 S-start / Random-start 평가 |
| **Library_env_test_advanced_image.py** | Sample Path 이미지 출력 |
| **Library_test_advanced_summary.py** | 전체 성공률/스텝/충돌/timeout 통계 |
| **library_AG_double_dueling.pt** | 최종 Double + Dueling 학습 모델 |

---

# Learning Strategy

### ✔ 단계적 Curriculum Learning  
| 단계 | 특징 |
|------|------|
| **Simple** | 구조 단순, 기본 이동 규칙 학습 |
| **Curriculum** | 장애물 + Random Start 비율 조절로 일반화 강화 |
| **Advanced** | 실제 환경 유사, 다중 타겟(A~G) 탐색 |

### ✔ Random Start 스케줄링  
- 초기: 100% 랜덤 시작 → 기본 탐색 패턴 습득  
- 중기: Random ↓ / S-start ↑ → 안정적 수렴  
- 후기: Random ↑(0→80%) → 일반화 강화  

### ✔ Reward 설계  
- 기본 이동: **–0.01 ~ –0.05**  
- 벽 충돌: **–1.0**  
- 목표 도달: **+10 ~ +20**  
→ 짧은 경로 유도 + 충돌 억제 + 안정적 탐색

---

# ▶ 실행 방법 (How to Run)

**1) Simple Training**
python ★Library_Simple/Library_train_random_start.py


**2) Curriculum Training**
python ★Library_Curriculum/train_curriculum_base.py

**3) Advanced Training**
python ★Library_Advanced/library_env_train.py

**4) Advanced Evaluation**
python ★Library_Advanced/Library_advanced_eval.py

**Key Results**

**✔ Seed 고정 vs Random Start 비교**

S-start 성공률: 83% ~ 88%

Random-start 성공률: 69% ~ 75%

95% CI: 69.8% ± 5.0%

→ seed 변화에도 안정적·일관된 행동 패턴 학습

Advanced 타겟(A~G)별 성능
Target	성공률
A	70%
B	66%
C	70%
D	66%
E	78%
F	90%
G	46%

※ G는 구조적으로 난이도 가장 높음 (좁은 복도 + 장애물)

**Sample Path Visualization**

(Advanced – Target G 예시)

<img width="850" src="https://github.com/user-attachments/assets/56a8e1e9-74c8-4d97-b192-507b209afb34" />

**Reproducibility
**
모든 환경에 시드 고정 가능

Curriculum/Advanced 단계의 통계 요약 스크립트 포함

완전한 재현을 위한 학습/평가 코드 제공

**Future Work
**
Multi-agent cooperative navigation

PPO/A3C 등 Actor-Critic 계열과 비교

실제 도서관 평면도 기반 시뮬레이션

LLM 기반 Reward shaping 실험


A71051 임재윤 (서강대학교 AI SW 대학원)
