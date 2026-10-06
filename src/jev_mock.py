"""
JEV Mock Data Interface
시뮬레이션 환경에서 배터리/이물질 투입 시 제어기로 주입할 데이터 인터페이스
"""

import json
from dataclasses import dataclass, asdict
from typing import List, Optional
from enum import Enum


class ItemType(Enum):
    """배터리/이물질 분류"""
    BATTERY_18650 = "18650"
    BATTERY_AA_AAA = "AA_AAA"
    AUX_PACK = "AUX_PACK"
    CAN_ALUMINUM = "CAN_AL"
    CAN_FERROUS = "CAN_FE"
    COIN_CR = "COIN_CR"
    COIN_SR = "COIN_SR"
    COIN_LR = "COIN_LR"
    UNKNOWN_WASTE = "UNKNOWN_WASTE"


class VisualStatus(Enum):
    """외관 상태"""
    NORMAL = "NORMAL"
    CORROSION = "CORROSION"
    SWELLING = "SWELLING"
    DAMAGED = "DAMAGED"


@dataclass
class JEVMockData:
    """JEV Mock 데이터 구조"""
    item_id: str
    item_type: ItemType
    visual_status: VisualStatus
    ocr_text: Optional[str] = None
    confidence: float = 0.95

    def to_json(self) -> str:
        """JSON 형식으로 변환"""
        data = {
            "item_id": self.item_id,
            "item_type": self.item_type.value,
            "visual_status": self.visual_status.value,
            "ocr_text": self.ocr_text,
            "confidence": self.confidence
        }
        return json.dumps(data, indent=2)

    def to_dict(self) -> dict:
        """딕셔너리 형식으로 변환"""
        return {
            "item_id": self.item_id,
            "item_type": self.item_type.value,
            "visual_status": self.visual_status.value,
            "ocr_text": self.ocr_text,
            "confidence": self.confidence
        }


class JEVMockFactory:
    """JEV Mock 데이터 생성 팩토리"""

    _counter = 0

    @classmethod
    def create_normal_battery(cls, battery_type: ItemType, ocr_text: Optional[str] = None) -> JEVMockData:
        """정상 배터리 생성"""
        cls._counter += 1
        return JEVMockData(
            item_id=f"ITEM_{2026}_01{cls._counter:02d}",
            item_type=battery_type,
            visual_status=VisualStatus.NORMAL,
            ocr_text=ocr_text
        )

    @classmethod
    def create_damaged_battery(cls, battery_type: ItemType, status: VisualStatus) -> JEVMockData:
        """손상된 배터리 생성"""
        cls._counter += 1
        return JEVMockData(
            item_id=f"ITEM_{2026}_01{cls._counter:02d}",
            item_type=battery_type,
            visual_status=status,
            confidence=0.88
        )

    @classmethod
    def create_foreign_object(cls, obj_type: ItemType) -> JEVMockData:
        """이물질 생성"""
        cls._counter += 1
        return JEVMockData(
            item_id=f"ITEM_{2026}_01{cls._counter:02d}",
            item_type=obj_type,
            visual_status=VisualStatus.NORMAL
        )

    @classmethod
    def reset_counter(cls):
        """카운터 리셋"""
        cls._counter = 0


class JEVProcessor:
    """JEV Mock 데이터 처리 및 의사결정 엔진"""

    @staticmethod
    def is_fire_hazard(jev_data: JEVMockData) -> bool:
        """화재 위험 판정 (외관 손상)"""
        return jev_data.visual_status in [VisualStatus.CORROSION, VisualStatus.SWELLING, VisualStatus.DAMAGED]

    @staticmethod
    def is_foreign_object(jev_data: JEVMockData) -> bool:
        """이물질 판정"""
        foreign_types = [
            ItemType.CAN_ALUMINUM,
            ItemType.CAN_FERROUS,
            ItemType.AUX_PACK,
            ItemType.UNKNOWN_WASTE
        ]
        return jev_data.item_type in foreign_types

    @staticmethod
    def is_coin_battery(jev_data: JEVMockData) -> bool:
        """코인 배터리 판정"""
        coin_types = [ItemType.COIN_CR, ItemType.COIN_SR, ItemType.COIN_LR]
        return jev_data.item_type in coin_types

    @staticmethod
    def get_destination(jev_data: JEVMockData) -> str:
        """투입물의 목표 위치 결정"""

        # 1차: 화재 위험 체크
        if JEVProcessor.is_fire_hazard(jev_data):
            return "isolation_bin"  # 소화 격리함

        # 2차: 이물질/보조배터리 체크
        if JEVProcessor.is_foreign_object(jev_data):
            return "manual_divert_zone"  # 수작업 선별칸

        # 3차: 코인 배터리는 정밀 파지 필요
        if JEVProcessor.is_coin_battery(jev_data):
            return "arm_a_diagnosis"  # Arm A로 정밀 진단

        # 4차: 일반 배터리는 펄스 부하 진단
        return "arm_a_diagnosis"


# 테스트 샘플 데이터
SAMPLE_WORKFLOW = [
    JEVMockFactory.create_normal_battery(ItemType.BATTERY_18650),
    JEVMockFactory.create_normal_battery(ItemType.COIN_CR, "CR2032"),
    JEVMockFactory.create_foreign_object(ItemType.CAN_ALUMINUM),
    JEVMockFactory.create_damaged_battery(ItemType.BATTERY_AA_AAA, VisualStatus.SWELLING),
    JEVMockFactory.create_normal_battery(ItemType.BATTERY_AA_AAA),
    JEVMockFactory.create_normal_battery(ItemType.COIN_SR, "SR626"),
]
