# Dual SO-101 Inverted Gantry 폐배터리 자동 선별 시스템

**프로젝트 개요**: Robofest Exhibition 참가용 폐배터리 및 이물질 자동 선별 시스템 시뮬레이션

## 🎯 시스템 구성

- **Dual Inverted Gantry**: 천장 매립형 듀얼 로봇 팔 (SO-101 x2)
- **엔드이펙터**: 3단 커스텀 핑거 (진단/정밀파지/고속이송)
- **분류 방식**: 
  1. 외관 검사 (화재 예방)
  2. 이물질 우회 (수작업 선별)
  3. 동적 펄스 부하 진단 (내부저항 R_int)
  4. 코인배터리 성분 분류 (CR/SR/LR)

## 📁 프로젝트 구조

```
da/
├── src/              # 핵심 제어 모듈
│   ├── jev_mock.py   # JEV Mock 데이터 인터페이스
│   ├── fsm_controller.py  # FSM 상태 제어
│   └── robot_kinematics.py  # IK 및 궤적 생성
├── simulation/       # PyBullet 시뮬레이션
│   └── main_simulation.py  # 메인 시뮬레이션 스크립트
└── config/          # 시스템 설정
    └── system_config.yaml
```

## 🚀 실행 방법

```bash
python simulation/main_simulation.py
```

## 📋 주요 기능

- [x] JEV Mock 데이터 주입
- [x] FSM 기반 상태 제어
- [x] Dual Gantry 배치 시뮬레이션
- [x] 충돌 없는 IK 기반 픽앤플레이스
- [ ] 0.1초 펄스 부하 진단 시뮬레이션
- [ ] 실시간 비전 카메라 뷰

---
**작성자**: Claude  
**버전**: 0.1.0-beta
