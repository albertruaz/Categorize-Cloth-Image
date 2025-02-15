PRIMARY_TO_SECONDARY = {
    1: [1, 2, 3, 4, 5, 24],        # 아우터
    2: [6, 7, 8, 9, 25, 26],       # 상의
    3: [10, 11, 12, 13, 14, 27],   # 바지
    4: [15, 28, 29],               # 원피스
    5: [17, 18, 19],               # 패션소품
    6: [20, 21, 22, 23],           # 가방
    7: [30, 16, 31],               # 스커트
    8: [32],                       # 셋업
    9: [33]                        # 신발
}

TRAIN_CONFIG = {
    'batch_size': 128,
    'num_workers': 2,
    'initial_lr': 1e-3,
    'fine_tune_lr': 1e-4,
    'fine_tune_epoch': 2,
    'total_epochs': 5
}

DATA_CONFIG = {
    'train_limit': 6000,
    'train_offset': 0,
    'val_limit': 1500,
    'val_offset': 6000
} 