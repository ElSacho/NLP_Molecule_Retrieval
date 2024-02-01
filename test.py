import tensorflow as tf

def extract_hyperparameters_from_event_file(event_file_path):
    raw_dataset = tf.data.TFRecordDataset(event_file_path)
    for raw_record in raw_dataset:
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        for v in example.features.feature['hparams'].bytes_list.value:
            print(v)

# Remplacez ceci par le chemin de votre fichier events.out.tfevents
event_file_path = 'glogs/loss/distil_both-2024-01-15--2340/events.out.tfevents.1705358406.jaguar.polytechnique.fr.1955901.0'
extract_hyperparameters_from_event_file(event_file_path)
