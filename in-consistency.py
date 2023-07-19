import numpy as np
import pandas as pd

image_caption = pd.read_csv('/tmp/pycharm_project_43/ImageBind-main/gener_files/Playing_With_Embeds/image_caption_distances.csv')
image_audio = pd.read_csv('/tmp/pycharm_project_43/ImageBind-main/gener_files/Playing_With_Embeds/image_audio_distances.csv')

in_consis = 2*image_caption - image_audio

in_consis.to_csv('in_consistency.csv', index=False)
