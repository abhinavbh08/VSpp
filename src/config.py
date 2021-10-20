# Dataset related config
root_path = "data/flickr30k_images"
images_folder = "flickr30k_images"
json_name = "dataset_flickr30k.json"
size = 224
models_folder = "models"

# Image Encoder config
model_name = "resnet"  # For now vgg and resnet are there.
normalise = True  # Whether to normlise embeddings obtained from the CNN
embedding_dim = (
    512  # This will be then compared with the embeddings obtained from the text model.
)
finetune = True

# Text Encoder config
word_embedding_dim=300
vocab_path = "models/vocab.pkl"

# Training config
device = "cpu"
batch_size=32
num_workers=4
learning_rate=0.0002
margin=0.2
num_epochs=5