# Dual SO-101 Inverted Gantry — 폐배터리/이물질 선별 시뮬레이션

천장 역거치 SO-101 두 대(실제 URDF/STL, TheRobotStudio/SO-ARM100)가 JEV Mock 판정에 따라 배터리/이물질을 분류한다.
팔 구동은 좌표 보간이 아니라 **PPO로 학습한 도달 정책**이 담당하고, FSM은 목표(어디로/어떤 파지)만 정한다.

## 구성

```
assets/so101/          실제 SO-101 URDF + STL (so101_grip.urdf = 조 패드 충돌을 박스로 단순화한 파지용)
src/robot.py           SO-101 래퍼(역거치, 헤딩, 토크 한계)
src/environment.py     셀 장면, 수거함 배치, 렌더링
src/rl_env.py          Gymnasium 도달 환경(목표 위치 + 목표 헤딩, 롤은 헤딩 비례 서보)
src/agent.py           학습된 정책 실행(안티와인드업, 자코비안 동기 서보, 보관 자세)
src/grasp.py           마찰 파지 계측, 코인용 자석 팁(외력 모델)
src/fsm.py             JEV 분류 -> 접근 -> 파지(접촉력 검증) -> 이송 -> 펄스 부하 -> 투입
src/jev_mock.py        JEV Mock 데이터 구조
simulation/train_reach.py   PPO 학습 (--hold --head --resume ...)
simulation/run_sim.py       시나리오 실행 + GIF/MP4 (--no-video 로 헤드리스)
simulation/eval_*.py        정책 정량 평가
models/                v8snap = 현재 사용 정책(헤딩 조건부), goals*.npy = 학습 목표 집합
```

## 실행

```bash
python3.11 -m venv venv && . venv/bin/activate        # pybullet 휠 문제로 3.11 권장
pip install -r requirements.txt
python simulation/run_sim.py --model models/v8snap_ppo --vecnorm models/v8snap_vecnorm.pkl --out out/sim.mp4
python simulation/run_sim.py --no-video --only 101,106,111   # 일부 아이템만
```

## 동작 요약

| 분류 | 처리 |
|---|---|
| 외관 손상(CORROSION/SWELLING/DAMAGED) | 팔 B가 소화 격리함으로 |
| 캔/보조배터리/미확인 이물질 | 양팔 협동(각 팔 닫힌 패드로 양쪽 압착, 접촉력 양쪽 검증) 후 수작업 선별칸으로 |
| 18650 / AA·AAA | 팔 A 핀치 파지 -> 0.1 s 펄스 부하로 R_int 산출 -> 재사용/재활용 |
| 코인(OCR) | 팔 A 자석 팁 -> CR/SR/LR 수거함, OCR 실패 시 수작업 |

파지는 구속(constraint) 없이 패드 마찰로만 한다. 실패하면 정상 분류로 세지 않고 FAULT로 기록한다(텔레포트 없음).

## 현재 결과 (14개 시나리오, 헤드리스 결정적 실행)

- 14개 중 **7개**가 목표 수거함에 들어감
- 잘 되는 것: 18650/AA 핀치 파지(접촉력 4~10 N, 파지 오차 약 3 mm), R_int 판정, 코인 자석 분류, 격리함 이송, 협동 파지 성립(양팔 12~14 N)과 큐브 이송
- 안 되는 것: 협동 이송의 신뢰성(두 팔 높이 불일치로 미끄러져 튕김), 납작한 보조배터리의 낮은 파지 높이(팔 A 손목 한계), 일부 18650 파지 재시도, 코인 CR 이송 중 분실
- 정책 정확도: 수거함 위치 도달 2~6 mm, 헤딩 오차 중앙값 0도 (`simulation/eval_head.py`)

## 알려진 한계

- 코인 자석은 외력 모델(질량 정규화 감쇠 스프링)이며 실제 자기장 모델이 아님
- 파지 마찰 계수는 안정적으로 잡히도록 높게 설정(패드 20, 대형 물체 10, 소형 3). 실제 고무 패드 값 아님
- 협동 압착은 위치 제어 팔의 강성 때문에 토크 한계(3 N·m)와 접촉력 피드백으로 제어
