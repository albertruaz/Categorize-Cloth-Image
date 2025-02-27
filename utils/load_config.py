import json


def load_config(config_path="config/config.json"):
    with open(config_path, "r") as file:
        config = json.load(file)

    # PRIMARY_TO_SECONDARY의 key를 int로 변환
    config["PRIMARY_TO_SECONDARY"] = {int(k): v for k, v in config["PRIMARY_TO_SECONDARY"].items()}
    
    # SECONDARY_TO_KOREAN의 key도 int로 변환
    config["SECONDARY_TO_KOREAN"] = {int(k): v for k, v in config["SECONDARY_TO_KOREAN"].items()}
    
    # 기존 unfreeze 처리
    config["unfreeze"] = {int(k): v for k, v in config["unfreeze"].items()}
    
    return config