from function_train import train
import sys

# list_config_path = ['configs/oneHead_linear_NLP.json', 'configs/oneHead.json', 'configs/baseline.json', 'configs/initVQ.json', 'configs/noMLP.json']
# list_config_path = ['configs/3head_cheb_fine_tune.json', 'configs/3head_conv_fine_tune.json','configs/3head_conv_fine_tune_VQ.json' ]
# list_config_path = ['configs/3head_conv_fine_tune_VQ.json' ]
# list_config_path = ['configs/1head_conv_fine_tune_VQ_simplified.json','configs/1head_conv_fine_tune_VQ_simplified_sci.json' ]
# list_config_path = ['configs/1head_scibert_no_accumulation_step.json', 'configs/1head_scibert_accumulation_step.json']
# list_config_path = ['configs/2MLP_linear_scibert.json']
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate.py <file_path>")
        sys.exit(1)

    # Le premier argument après le nom du script est le file_path
    file_path = sys.argv[1]

    list_config_path = [file_path]
    train(list_config_path)