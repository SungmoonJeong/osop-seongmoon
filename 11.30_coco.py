from pycocotools.coco import COCO

# COCO 데이터셋 경로 설정 (annotations 및 images 폴더가 있는 디렉토리)
dataDir = 'C:\\Users\\gjaischool\\Desktop\\osop\\datasets\\'
dataType = 'train2017'  # 데이터셋 유형 선택 (train, val, test)

# COCO 데이터셋 객체 생성
annFile = f'{dataDir}/annotations/instances_{dataType}.json'
coco = COCO(annFile)

# 데이터셋 정보 출력
categories = coco.loadCats(coco.getCatIds())
print(f'카테고리 개수: {len(categories)}')

imgIds = coco.getImgIds()
print(f'이미지 개수: {len(imgIds)}')

annIds = coco.getAnnIds()
print(f'어노테이션 개수: {len(annIds)}')

# 랜덤 이미지와 어노테이션 가져오기
import random

imgId = random.choice(imgIds)
img = coco.loadImgs(imgId)[0]
annIds = coco.getAnnIds(imgIds=img['id'])
anns = coco.loadAnns(annIds)

# 이미지와 어노테이션 시각화 (예시로 matplotlib 사용)
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

image_path = f'{dataDir}/images/{dataType}/{img["file_name"]}'
image = plt.imread(image_path)
plt.imshow(image)

for ann in anns:
    bbox = ann['bbox']
    category = categories[ann['category_id']]['name']
    plt.gca().add_patch(Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=2, edgecolor='r', facecolor='none'))
    plt.text(bbox[0], bbox[1] - 5, category, color='r', fontsize=12)

plt.axis('off')
plt.show()
