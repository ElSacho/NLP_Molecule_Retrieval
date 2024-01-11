from models.oneHead_LinearNLP import TextEncoder

parameters = {'model_name': 'allenai/scibert_scivocab_uncased', "nout": 768}
model = TextEncoder(parameters)
print(model)