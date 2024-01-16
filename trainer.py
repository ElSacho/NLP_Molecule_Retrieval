from function_train import train
import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate.py <file_path>")
        sys.exit(1)

    # Le premier argument après le nom du script est le file_path
    file_path = sys.argv[1]

    list_config_path = [file_path]
    train(list_config_path)