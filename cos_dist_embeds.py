# COSINE-DISTANCE COMPUTANTION BETWEEN IMAGE-TEXT AND IMAGE-AUDIO EMBEDDINGS

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances
from scipy.spatial.distance import cosine

# Load IMAGE embeddings from CSV file (Change this for your path to the CSV file of the image embeddings)
image_embeddings = pd.read_csv('/tmp/pycharm_project_43/ImageBind-main/gener_files/csv_embeds/image_600_embeds_imagebind.csv').values
# Load CAPTION embeddings from CSV file (Change this for your path to the CSV file of the caption embeddings)
caption_embeddings = pd.read_csv('/tmp/pycharm_project_43/ImageBind-main/gener_files/csv_embeds/text_12k_embeds_imagebind.csv').values
# Load AUDIO embeddings from CSV file (Change this for your path to the CSV file of the audio embeddings)
audio_embeddings = pd.read_csv('/tmp/pycharm_project_43/ImageBind-main/gener_files/csv_embeds/audio_embeds2.csv').values

# Compute the number of images
num_images = image_embeddings.shape[0]

# Initialize arrays to store distances
image_caption_distances = np.zeros((num_images, 20))
image_audio_distances = np.zeros((num_images, 20))

# Compute cosine distances between image and captions
for i in range(num_images):
    image_emb = image_embeddings[i]
    for j in range(20):
        caption_emb = caption_embeddings[i * 20 + j]
        distance = cosine(image_emb, caption_emb)
        image_caption_distances[i, j] = distance
        print(caption_emb.shape)

# Compute cosine distances between image and audios
for i in range(num_images):
    image_emb = image_embeddings[i]
    for j in range(20):
        audio_emb = audio_embeddings[i * 20 + j]
        distance = cosine(image_emb, audio_emb)
        image_audio_distances[i, j] = distance

# Save image-caption distances to CSV file (Select the name you want to give to your .csv file for image-text distance)
image_caption_df = pd.DataFrame(image_caption_distances)
image_caption_df.to_csv('image_caption_distances2.csv', index=False)

# Save image-audio distances to CSV file (Select the name you want to give to your .csv file for image-audio distance)
image_audio_df = pd.DataFrame(image_audio_distances)
image_audio_df.to_csv('image_audio_distances2.csv', index=False)