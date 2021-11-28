# Dataset related config
# root_path = "data/flickr30k_images"
# images_folder = "flickr30k_images"
# json_name = "dataset_flickr30k.json"

root_path = ""
images_folder = "/kaggle/input/flickr-image-dataset/flickr30k_images/flickr30k_images"
json_name = "/kaggle/input/captions/dataset_flickr30k.json"
size = 224
models_folder = "models"

# Image Encoder config
model_name = "vgg"  # For now vgg and resnet are there.
normalise = False  # Whether to normlise embeddings obtained from the CNN
embedding_dim = (
    1024  # This will be then compared with the embeddings obtained from the text model.
)
finetune = True
pretrained = True

# Text Encoder config
word_embedding_dim = 300
vocab_path = "models/vocab.pkl"

# Training config
batch_size = 50
num_workers = 4
learning_rate = 0.0002
margin = 0.2
num_epochs = 50