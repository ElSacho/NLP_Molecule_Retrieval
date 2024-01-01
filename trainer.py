from function_train import train

# list_config_path = ['configs/oneHead_linear_NLP.json', 'configs/oneHead.json', 'configs/baseline.json', 'configs/initVQ.json', 'configs/noMLP.json']
list_config_path = ['configs/3head_cheb_fine_tune.json', 'configs/3head_conv_fine_tune.json','configs/3head_conv_fine_tune_VQ.json' ]

train(list_config_path)