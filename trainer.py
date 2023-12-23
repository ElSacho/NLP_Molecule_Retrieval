from train import train

list_config_path = ['attentionOnGraph/configs.json']

for config_path in list_config_path:
    train(config_path)