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

def load_image(image_url, max_retries=3, timeout=10):
    """이미지를 로드하고 PIL Image 객체로 반환한다. 실패 시 지정된 횟수만큼 재시도한다."""
    headers = {
        "User-Agent": "Vingle-AI-Train-Category"
    }
    for attempt in range(1, max_retries + 1):
        try:
            # 타임아웃 설정 (초 단위)
            response = requests.get(image_url, headers=headers, timeout=timeout)

            # HTTP 에러가 발생했을 경우 예외 발생
            response.raise_for_status()

            # 이미지를 메모리에 로드
            img = Image.open(BytesIO(response.content)).convert("RGB")
            return img

        except Exception as e:

            # print(f"[{attempt}/{max_retries}] Error loading image from {image_url}: {e}")
            if attempt == max_retries:
                return None #Image.new('RGB', (224, 224), 'black')
            
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
        self.data = []
        for row in data_rows:
            img = load_image(row['image_url'])
            if img is not None:  # None이 아닐 때만 추가
                self.data.append(row)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int, int]:
        row = self.data[idx]
        image_url = row['image_url']
        product_id = row['id']
        
        img = load_image(image_url)
        if img is None:
            raise ValueError(f"Unexpected None image at index {idx}")

        # 이미지 변환 적용
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        # 레이블 생성
        pid = row['primary_category_id']
        sid = row['secondary_category_id']

        return img, product_id, sid


