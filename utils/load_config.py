import json


def load_config(config_path="config/config.json"):
    with open(config_path, "r") as file:
        config = json.load(file)

    # PRIMARY_TO_SECONDARY의 key를 int로 변환
    config["PRIMARY_TO_SECONDARY"] = {int(k): v for k, v in config["PRIMARY_TO_SECONDARY"].items()}
    config["unfreeze_schedule"] = {int(k): v for k, v in config["unfreeze_schedule"].items()}
    
    return config