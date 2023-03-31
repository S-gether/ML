# ==============================================================
# 0. 라이브러리 불러오기
# ==============================================================
from collections import deque
import time
import random

# 화면의 너비, 높이 선언
width, height = 600, 480

# 사람 집중도 판정을 위한 위치 변화 거리 기준 지정
con_min_distance = 0.01
con_max_distance = 0.2
# 사람 중심 좌표 저장할 Queue 생성
person_center_queue = deque()
# person의 현재 중점 좌표 비율 임시 저장
temp_center_ratio_x, temp_center_ratio_y = 0, 0
print(person_center_queue)
while(True):
    # 임시용!!!
    time.sleep(1)
    # 결과 데이터 임시로 생성 !!!
    det = []
    # 사람 확률 80% 휴대폰 확률 20% 물체 개수 1~3개 생성 !!!
    for i in range(random.randint(1, 4)):
        x1, y1 = random.randint(0, width), random.randint(0, height)
        x2, y2 = random.randint(x1, width), random.randint(y1, height)
        conf, cls = 0.4, random.choice(["0", "67", "0", "0", "0"])
        det.append([x1, y1, x2, y2, conf, cls])

    # 결과 데이터 중에서 person인 경우 각 bbox의 면적 추출
    person_bbox_area = [(ob[2] - ob[0]) * (ob[3] - ob[1]) for idx, ob in enumerate(det) if ob[5] == "0"]
    # 결과 데이터 중에서 phone인 경우 각 bbox의 면적 추출
    phone_bbox_area = [(ob[2] - ob[0]) * (ob[3] - ob[1]) for idx, ob in enumerate(det) if ob[5] == "67"]

    # 사람이 존재 하는 경우
    if len(person_bbox_area) >= 1:
        # 휴대폰이 존재하지 않는 경우
        if len(phone_bbox_area) == 0:
            # bbox 중에서 가장 면적이 큰 사람의 인덱스 추출
            person_idx = person_bbox_area.index(max(person_bbox_area))
            # person의 중점 좌표 추출
            person_center_x, person_center_y = det[person_idx][0] + (det[person_idx][2] - det[person_idx][0]) / 2, det[person_idx][1] + (det[person_idx][3] - det[person_idx][1]) / 2
        # 휴대폰 존재하는 경우
        else:
            concentration = 0
            print(f"concentration {concentration} | person_bbox_area {person_bbox_area} | phone_bbox_area {phone_bbox_area}")
            print(f"queue_len {len(person_center_queue)} queue {person_center_queue}\n")
            continue
    # 사람이 존재하지 않는 경우
    else:
        concentration = 0
        print(f"concentration {concentration} | person_bbox_area {person_bbox_area} | phone_bbox_area {phone_bbox_area}")
        print(f"queue_len {len(person_center_queue)} queue {person_center_queue}\n")
        continue

    # person의 현재 중점 좌표 비율과 이전 중점 좌표 비율 사이의 거리 측정
    person_center_queue.append(((temp_center_ratio_x - (person_center_x / width))**2 + (temp_center_ratio_x - (person_center_y / height))**2)**0.5)
    # person의 중점 좌표가 20개 이상이 쌓이면
    if (len(person_center_queue) > 20):  # 주기가 3초라고 가정했을 때, 60초 지난 시점의 좌표는 버려
        # 오래된 좌표 1개 제거
        person_center_queue.popleft()

        # 20개 데이터 전부 움직임 비율이 con_min_distance 이하면, 집중 안한 것(잠듬)
        if len(list(filter(lambda x: x < con_min_distance, person_center_queue))) == 20:
            concentration = 0
        # 20개 데이터 중 움직임 비율이 con_max_distance 이상인 데이터가 10개 이상이면, 집중 안한 것(몸을 너무 흔들정도의 산만함)
        elif len(list(filter(lambda x: x > con_max_distance, person_center_queue))) >= 10:
            concentration = 0
        # 20개 데이터 중 하나라도 움직임 비율이 con_min_distance 이상이고 10개 미만의 데이터의 움직임 비율이 con_max_distance 이하이면, 집중을 한 것
        else:
            concentration = 1
    else:
        concentration = 2 # 이거 대기시간으로 바꿔야할듯???

    # person의 현재 중점 좌표 비율 임시 저장
    temp_center_ratio_x, temp_center_ratio_y = person_center_x / width, person_center_y / width

    print(f"concentration {concentration} | person_bbox_area {person_bbox_area} | phone_bbox_area {phone_bbox_area}")
    print(f"queue_len {len(person_center_queue)} queue {person_center_queue}\n")

"""
im.shape : (1, 3, 480, 640)
im0s.shape : (1, 480, 640, 3)

scale 바꾸기

pred.shape : (7, 6)
- 행 : 각 물체
- 열 : x1, y1, x2, y2, confidence_score, class 종류

- class 종류
    - 67 : 휴대폰
    - 0 : 사람
    
im.shape[2:] : (480, 640)
im0.shape : (480, 640, 3)

gain : min(im.shape[2:][0] / im0.shape[0], im.shape[2:][1] / im0.shape[1])
pad = (im.shape[2:][1] - im0.shape[1] * gain) / 2, (im.shape[2:][0] - im0.shape[0] * gain) / 2
    
휴대폰과 사람 같이 있으면 -> 집중 안함
사람만 있으면 -> 집중함
사람 좌표가 일정 시간 이상 고정되어있으면 -> 집중 안함
        
사람이 감지되는 경우
    사람이 2명 이상인 경우
        둘 중 더 큰 면적을 가지는 바운딩 박스의 중점을 보겠다
    사람이 1명인 경우
        바운딩 박스의 중점을 보겠다
    해당 바운딩 박스의 중점을 Queue에 저장함
    Queue에서 원소 10개 이상이면 가장 오래된거 버림 (주기가 3초면 30초 지난값은 버린다는 뜻)
    Queue에서 가장 오래된 값과 현재 값을 비교해서 거리를 측정함. 일정 이상의 거리가 나오지 않으면 집중하지 않은 것임
    일정 이상 거리 : 집중함
    일정 이하 거리 : 집중하지 않음
휴대폰이 감지되는 경우
    집중하지 않음
그 외
    집중하지 않음
    
# 모의고사와 같이 휴대폰을 보면 안되는 상황 가정
# 사람이 너무 작은 경우 집중 안함 판정
# 거리가 너무 많이 차이나도 집중 안하는걸로 판정
"""
