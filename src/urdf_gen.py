"""SO-101 URDF의 그리퍼 충돌 형상을 단순 박스로 교체한 마찰 파지용 URDF 생성.
(원본 STL 메시는 동적 바디에서 볼록 껍질로 처리되어 조 사이 틈이 사라지므로 핑거 패드만 박스로 대체. 시각 메시는 원본 유지)"""

import os
import re

import numpy as np
from scipy.spatial.transform import Rotation as R

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "so101")
SRC = os.path.join(ROOT, "so101_new_calib.urdf")
DST = os.path.join(ROOT, "so101_grip.urdf")

PAD_W, PAD_T, PAD_Z0, PAD_Z1 = 0.024, 0.015, -0.040, 0.006   # 패드 폭(y), 두께(x), 툴 z 범위
FIXED_FACE_X = 0.002        # 고정 핑거 안쪽 면 (툴 프레임 x)
JAW_FACE_X_CLOSED = -0.018  # 이동 조 안쪽 면 (q=0)


def _T(xyz, rpy):
    M = np.eye(4)
    M[:3, :3] = R.from_euler("xyz", rpy).as_matrix()
    M[:3, 3] = xyz
    return M


def _box_xml(M, size):
    rpy = R.from_matrix(M[:3, :3]).as_euler("xyz")
    return (f'<collision><origin xyz="{M[0,3]:.6f} {M[1,3]:.6f} {M[2,3]:.6f}" rpy="{rpy[0]:.6f} {rpy[1]:.6f} {rpy[2]:.6f}"/>'
            f'<geometry><box size="{size[0]:.5f} {size[1]:.5f} {size[2]:.5f}"/></geometry></collision>')


def build():
    txt = open(SRC).read()
    t_tool = _T([-0.0079, -0.000218121, -0.0981274], [0, 3.14159, 0])       # gripper_link 내 툴 프레임
    t_jaw = _T([0.0202, 0.0188, -0.0234], [1.5708, -5.24284e-08, -1.41553e-15])  # gripper_link 내 조 관절(q=0)
    zc, zh = (PAD_Z0 + PAD_Z1) / 2, PAD_Z1 - PAD_Z0
    c_fixed = _T([FIXED_FACE_X + PAD_T / 2, 0, zc], [0, 0, 0])
    c_jaw = _T([JAW_FACE_X_CLOSED - PAD_T / 2, 0, zc], [0, 0, 0])
    fixed_box = _box_xml(t_tool @ c_fixed, (PAD_T, PAD_W, zh))
    jaw_box = _box_xml(np.linalg.inv(t_jaw) @ t_tool @ c_jaw, (PAD_T, PAD_W, zh))

    def swap(link_name, mesh, new_collision):
        nonlocal txt
        m = re.search(rf'<link name="{link_name}">.*?</link>', txt, re.S)
        block = m.group(0)
        pat = re.compile(r'<collision>(?:(?!</collision>).)*?' + re.escape(mesh) + r'.*?</collision>', re.S)
        assert pat.search(block), (link_name, mesh)
        txt = txt.replace(block, pat.sub(new_collision, block))

    swap("gripper_link", "wrist_roll_follower_so101_v1.stl", fixed_box)
    swap("moving_jaw_so101_v1_link", "moving_jaw_so101_v1.stl", jaw_box)
    open(DST, "w").write(txt)
    return DST


if __name__ == "__main__":
    print(build())
