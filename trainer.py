from function_train import train

# list_config_path = ['configs/oneHead_linear_NLP.json', 'configs/oneHead.json', 'configs/baseline.json', 'configs/initVQ.json', 'configs/noMLP.json']
list_config_path = ['configs/baseline.json', 'configs/baseline_VQ.json']

train(list_config_path)