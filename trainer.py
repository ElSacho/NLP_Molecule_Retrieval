from function_train import train

# list_config_path = ['configs/oneHead_linear_NLP.json', 'configs/oneHead.json', 'configs/baseline.json', 'configs/initVQ.json', 'configs/noMLP.json']
# list_config_path = ['configs/3head_cheb_fine_tune.json', 'configs/3head_conv_fine_tune.json','configs/3head_conv_fine_tune_VQ.json' ]
# list_config_path = ['configs/3head_conv_fine_tune_VQ.json' ]
# list_config_path = ['configs/1head_conv_fine_tune_VQ_simplified.json','configs/1head_conv_fine_tune_VQ_simplified_sci.json' ]
# list_config_path = ['configs/1head_scibert_no_accumulation_step.json', 'configs/1head_scibert_accumulation_step.json']
# list_config_path = ['configs/2MLP_linear_scibert.json']
list_config_path = ['configs/2MLP_linear_scibert.json']
train(list_config_path)