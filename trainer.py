from function_train import train

list_config_path = ['configs/baseline.json', 'configs/initVQ.json', 'configs/noMLP.json', 'configs/oneHead_linear_NLP.json', 'configs/oneHead.json']

train(list_config_path)