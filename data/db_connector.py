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

import random

from collections import defaultdict

def split_train_val(datas: list, train_ratio: float = 0.8):
    """
    datas를 secondary_category_id별로 묶어,
    각 카테고리에서 train_ratio 비율로 train과 val로 나눈다.
    """
    
    # 카테고리별로 묶기
    grouped = defaultdict(list)
    for d in datas:
        grouped[d['secondary_category_id']].append(d)

    train_rows = []
    val_rows = []
    num_for_class = [None for _ in range(33)]
    train_num_for_class = [None for _ in range(33)]
    val_num_for_class = [None for _ in range(33)]

    for category_id, items in grouped.items():
        random.shuffle(items)
        cat_count = len(items)
        # train_ratio(4:1 => 0.8)만큼 train에 할당
        train_count = int(cat_count * train_ratio)

        # 순서를 유지한 채로 앞부분은 train, 뒷부분은 val
        train_items = items[:train_count]
        val_items = items[train_count:]

        train_rows.extend(train_items)
        val_rows.extend(val_items)

        num_for_class[category_id-1] = cat_count
        train_num_for_class[category_id-1] = len(train_items)
        val_num_for_class[category_id-1] = len(val_items)


    return train_rows, val_rows, num_for_class, train_num_for_class, val_num_for_class

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

    def get_product_data(self, where_condition: str = "1=1", x: int = 10, offset: int = 0) -> list:
        """
        secondary_category_id가 같을 때, 해당 카테고리에 속하는
        데이터 개수가 x개 이상이면 x개만, x개 이하이면 전부 가져온다.
        """
        session = self.Session()
        try:
            # Window Function을 사용하여 secondary_category_id별로 데이터 개수(cat_count)를 구하고,
            # 같은 카테고리에 속하는 row들에 대한 순번(rownum)을 매긴다.
            # 그리고 rownum <= LEAST(cat_count, :x)를 만족하는 데이터만 조회한다.
            sql = text(f"""
                SELECT
                    id,
                    main_image,
                    status,
                    primary_category_id,
                    secondary_category_id
                FROM (
                    SELECT
                        id,
                        main_image,
                        status,
                        primary_category_id,
                        secondary_category_id,
                        -- secondary_category_id별 총 개수
                        COUNT(*) OVER (PARTITION BY secondary_category_id) AS cat_count,
                        -- secondary_category_id별 순번
                        ROW_NUMBER() OVER (PARTITION BY secondary_category_id ORDER BY id DESC) AS rownum
                    FROM product
                    WHERE {where_condition}
                ) AS t
                -- 각 secondary_category_id 그룹에서 cat_count와 x 중 더 작은 값까지만 가져온다.
                WHERE t.rownum <= LEAST(t.cat_count, :x)
                ORDER BY t.secondary_category_id, t.id DESC
            """)

            # 파라미터 바인딩
            result = session.execute(sql, {'x': x})

            products = []
            for row in result.fetchall():
                product_id = row["id"]
                main_image = self.get_s3_url(row["main_image"]) if row["main_image"] else None
                status = row["status"]
                primary_id = row["primary_category_id"]
                secondary_id = row["secondary_category_id"]

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