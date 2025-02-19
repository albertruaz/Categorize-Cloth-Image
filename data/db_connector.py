import torch
from torch.utils.data import Dataset, DataLoader
import requests
from PIL import Image
from io import BytesIO
from torchvision import transforms
from typing import List, Dict, Optional, Tuple

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sshtunnel import SSHTunnelForwarder
from typing import List, Dict
import os
import json
from dotenv import load_dotenv
import time


load_dotenv("info/.env")

# 카테고리 매핑 정의
# PRIMARY_TO_SECONDARY = {
#     1: [1, 2, 3, 4, 5, 24],        # 아우터
#     2: [6, 7, 8, 9, 25, 26],       # 상의
#     3: [10, 11, 12, 13, 14, 27],   # 바지
#     4: [15, 28, 29],               # 원피스
#     5: [17, 18, 19],               # 패션소품
#     6: [20, 21, 22, 23],           # 가방
#     7: [30, 16, 31],               # 스커트
#     8: [32],                       # 셋업
#     9: [33]                        # 신발
# }

def load_image(image_url, max_retries=3, timeout=10):
    """이미지를 로드하고 PIL Image 객체로 반환한다. 실패 시 지정된 횟수만큼 재시도한다."""
    for attempt in range(1, max_retries + 1):
        try:
            # 타임아웃 설정 (초 단위)
            response = requests.get(image_url, timeout=timeout)

            # HTTP 에러가 발생했을 경우 예외 발생
            response.raise_for_status()

            # 이미지를 메모리에 로드
            img = Image.open(BytesIO(response.content)).convert("RGB")
            return img

        except Exception as e:
            print(f"[{attempt}/{max_retries}] Error loading image from {image_url}: {e}")
            
            # 재시도 횟수를 모두 소진했다면, 빈(검정) 이미지로 대체
            if attempt == max_retries:
                return Image.new('RGB', (224, 224), 'black')
            
            # 잠시 대기 후 재시도
            time.sleep(1)

class ProductImageDataset(Dataset):
    def __init__(self, data_rows: List[Dict], transform: Optional[transforms.Compose] = None):
        """
        제품 이미지 데이터셋 초기화
        
        Args:
            data_rows: DB에서 가져온 제품 데이터 리스트
            transform: 이미지 변환을 위한 torchvision transforms
        """
        super().__init__()
        self.transform = transform

        # 유효한 데이터만 필터링
        valid_rows = []
        for row in data_rows:
            pid = row.get('primary_category_id')
            sid = row.get('secondary_category_id')
            img_url = row.get('image_url')

            # null check 및 유효한 카테고리 조합 확인
            # if (pid is None or sid is None or img_url is None or
            #     pid not in PRIMARY_TO_SECONDARY or
            #     sid not in PRIMARY_TO_SECONDARY[pid]):
            #     continue
                
            valid_rows.append(row)

        self.data = valid_rows

        # Primary 카테고리 ID를 0~N-1로 매핑
        # primary_ids = sorted(list(set(d['primary_category_id'] for d in self.data)))
        # self.primary_to_idx = {pid: i for i, pid in enumerate(primary_ids)}
        # self.num_primary_classes = len(primary_ids)

        # Secondary 카테고리 매핑 생성
        # self.secondary_mapping = {}
        # for pid in primary_ids:
        #     valid_subcats = PRIMARY_TO_SECONDARY[pid]
        #     self.secondary_mapping[pid] = {
        #         'subcats': valid_subcats,
        #         'subcat_to_idx': {sc: idx for idx, sc in enumerate(valid_subcats)}
        #     }

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int, int]:
        row = self.data[idx]
        image_url = row['image_url']
        product_id = row['id']
        
        img = load_image(image_url)

        # 이미지 변환 적용
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        # 레이블 생성
        pid = row['primary_category_id']
        sid = row['secondary_category_id']

        # primary_label = self.primary_to_idx[pid]
        # sub_info = self.secondary_mapping[pid]
        # secondary_label_local = sub_info['subcat_to_idx'][sid]
        
        return img, product_id, sid
        # return img, primary_label, pid, secondary_label_local, product_id

class SingletonMeta(type):
    _instance = None
    def __call__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__call__(*args, **kwargs)
        return cls._instance

class DBConnector(metaclass=SingletonMeta):
    def __init__(self):
        if hasattr(self, 'engine') and self.engine is not None:
            return  # 이미 초기화되었다면 재호출 방지

        self.ssh_host = os.getenv('SSH_HOST')
        self.ssh_username = os.getenv('SSH_USERNAME')
        self.ssh_pkey_path = os.getenv('SSH_PKEY_PATH')
        self.remote_bind_address = (
            os.getenv('DB_HOST'),
            int(os.getenv('DB_PORT', 3306))
        )

        self.db_user = os.getenv('DB_USER')
        self.db_password = os.getenv('DB_PASSWORD')
        self.db_name = os.getenv('DB_NAME')

        self.pool_size = int(os.getenv('DB_POOL_SIZE', 10))
        self.max_overflow = int(os.getenv('DB_MAX_OVERFLOW', 20))
        self.pool_timeout = int(os.getenv('DB_POOL_TIMEOUT', 30))
        self.pool_recycle = int(os.getenv('DB_POOL_RECYCLE', 3600))

        self.tunnel = None
        self.engine = None
        self.Session = None
        self.connect()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def connect(self):
        if self.tunnel is not None and self.tunnel.is_active:
            return
        self.tunnel = SSHTunnelForwarder(
            (self.ssh_host, 22),
            ssh_username=self.ssh_username,
            ssh_pkey=self.ssh_pkey_path,
            remote_bind_address=self.remote_bind_address
        )
        self.tunnel.start()

        db_url = f"mysql+pymysql://{self.db_user}:{self.db_password}@127.0.0.1:{self.tunnel.local_bind_port}/{self.db_name}"
        self.engine = create_engine(
            db_url,
            pool_size=self.pool_size,
            max_overflow=self.max_overflow,
            pool_timeout=self.pool_timeout,
            pool_recycle=self.pool_recycle
        )
        self.Session = sessionmaker(bind=self.engine)

    def close(self):
        if self.Session:
            self.Session.close_all()
            self.Session = None
        if self.tunnel and self.tunnel.is_active:
            self.tunnel.close()
        self.tunnel = None
        self.engine = None
        DBConnector._instance = None

    def get_s3_url(self, file_name: str) -> str:
        """ .env의 S3_CLOUDFRONT_DOMAIN 에 맞춰 S3 url 생성 """
        cloudfront_domain = os.getenv('S3_CLOUDFRONT_DOMAIN')
        protocol = "https"
        if not file_name or not cloudfront_domain:
            return None
        return f"{protocol}://{cloudfront_domain}/{file_name}"

    def get_product_data(self, where_condition: str = "1=1", limit: int = 500, offset: int = 0) -> list:
        """ 샘플로 특정 조건과 limit, offset을 기반으로 제품 정보를 가져온다. """
        session = self.Session()
        try:
            sql = text(f"""
                SELECT
                    id,
                    main_image,
                    status,
                    primary_category_id,
                    secondary_category_id
                FROM product
                WHERE {where_condition}
                LIMIT {limit} OFFSET {offset}
            """)
            result = session.execute(sql)

            products = []
            for row in result.fetchall():
                product_id = row[0]
                main_image = self.get_s3_url(row[1]) if row[1] else None
                status = row[2]
                primary_id = row[3]
                secondary_id = row[4]

                products.append({
                    'id': product_id,
                    'image_url': main_image,
                    'status': status,
                    'primary_category_id': primary_id,
                    'secondary_category_id': secondary_id
                })
            return products
        finally:
            session.close()
