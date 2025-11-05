from nanogpt_utils import *

######################
# plotting paradigm
######################
import torch
import torch.nn as nn
from torch.nn import functional as F
import requests
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
import platform
from tokenizers.processors import TemplateProcessing
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from matplotlib import font_manager
import numpy as np
import os
from PIL import Image

base_dirs = ['/Users/abdelrahmansawalma/Downloads/LLMs',
             r"\\klinik.uni-wuerzburg.de\homedir\userdata11\Sawalma_A\data\downloads\LLMs"]
base_dir = [item for item in base_dirs if os.path.exists(item)][0]


Alice_in_Wonderland = requests.get('https://www.gutenberg.org/cache/epub/11/pg11.txt').text.replace('\r','').replace('\n\n','\n')
Alice_in_Wonderland = Alice_in_Wonderland[Alice_in_Wonderland.find('CHAPTER I.\nDown the Rabbit-Hole')::]
encode, decode, tokenizer, vocab_size = tokenize(Alice_in_Wonderland, 6000) # decreasing vocab size tends to give smaller and smaller tokens (to the letter level)






# split by tokens and give back the token positions in a plot
def tkn_pos_finder(text, spacing = 0.05):
    '''
    returns the positions of tokens in a sentence, as well as the tokens and token IDs
    '''
    text = ' '.join(text.split(' ')[0:10]) #make sure there is no more than 10 words
    encoded = tokenizer.encode(text)
    good_inds = [i for i in range(len(encoded.tokens)) if encoded.tokens[i] not in ['<EOS>','<BOS>']]
    token_ids = np.array(encoded.ids)[good_inds]
    tokens = [tokenizer.decode([item]) for item in token_ids]

    x_pos = [0]+np.cumsum([len(token) * 0.12+spacing for token in tokens]).tolist()
    x_widths = [x_pos[i+1] - x_pos[i]-spacing for i in range(len(x_pos)-1)]
    x_pos = x_pos[0:-1]
    return tokens, token_ids, x_pos, x_widths

########################
# plot tokens
########################
def plot_line_tokens(text, ax, colors = ['#e6f3ff', '#fff0e6', '#e6ffe6', '#f0e6ff', '#fffae6'],
                     y_height=0.8, plot_ids = True, y_pos=0, spacing = 0):
    tokens, token_ids, x_pos, x_widths = tkn_pos_finder(text, spacing=spacing)

    for i, (token, token_id,x_p, x_w) in enumerate(zip(tokens, token_ids, x_pos,x_widths)):
        rect = patches.Rectangle((x_p, y_pos), x_w, y_height,linewidth=1.5, edgecolor='black',
                                 facecolor=colors[i % len(colors)], alpha=0.7)
        ax.add_patch(rect)

        # Add token text
        ax.text(x_p + x_w / 2, y_pos + y_height / 2,token, ha='center', va='center', fontsize=10,
                fontfamily='monospace', weight='bold')

        if plot_ids:
            ax.text(x_p + x_w / 2, y_pos - 0.2, f'{token_id}', ha='center', va='center', fontsize=8,
                    fontfamily='monospace', color='gray')
    max_x = x_p + x_w # to find the maximum x position for that line
    return max_x

text = 'Alice in Wonderland. Down the rabbit hole'
fig, ax = plt.subplots( figsize=(7,1.5))
max_x = plot_line_tokens(text, ax, spacing = 0.05)
ax.set_title('Tokenization', fontsize=18, fontfamily='Calibri')
ax.text(-0.1, 0.4, 'Tokens: ', ha='right', va='center', fontsize=14, fontfamily='Calibri', weight='bold')
ax.text(-0.1, -0.2 , 'IDs: ', ha='right', va='center', fontsize=14, color ='grey', fontfamily='Calibri', weight='bold')
ax.set_xlim(-0.7, max_x + 0.1)
ax.set_ylim(-0.1, 1.5)
ax.set_aspect(0.6)
ax.axis('off')  # Turn off axes
plt.tight_layout()
fig.savefig(os.path.join(os.path.join(base_dir,"figures"), 'tokenization.png'), dpi = 300)

# plot the progress of prediction
texts = 'To be, or not to be. That is the question.\nWhether it is nobler in the mind to suffer.\nThe slings and arrows of outrageous fortune.\nOr to take arms against a sea of troubles.'
texts = texts.split('\n')

figsize = (7,4)
y_height = 0.9
y_pos = [i for i in range(len(texts))][::-1]
colors = ['white']
max_x_final = 0
fig, ax = plt.subplots( figsize=figsize)
for text,y_p in zip(texts,y_pos):
    max_x = plot_line_tokens(text, ax,y_height = y_height, plot_ids=False, colors = colors, y_pos=y_p)
    max_x_final = np.max([max_x_final, max_x])
ax.set_xlim(-0.1, max_x_final + 0.1)
ax.set_ylim(-4, y_pos[0]+y_height+0.1)
ax.set_aspect(0.3)
ax.axis('off')  # Turn off axes
ax.text(0.75, -2.8 + y_height, '  Input: ', ha='left', va='bottom', fontsize=15, fontfamily='Calibri', weight='bold')
ax.text(3.75, -2.8 + y_height, '  Prediction: ', ha='left', va='bottom', fontsize=15, fontfamily='Calibri', weight='bold')
ax.text(0.5, -3 + y_height / 2, 'Tokens: ', ha='right', va='center', fontsize=12, fontfamily='Calibri', weight='bold')
ax.text(0.5, -3.75 , 'ID: ', ha='right', va='center', fontsize=12, fontfamily='Calibri', weight='bold')
plt.tight_layout()

ax.set_title('Tokenization and Encoding\n', fontsize=18, fontfamily='Calibri', weight='bold')

tokenized = [tkn_pos_finder(text,0) for text in texts]
x_tokens = [item[0] for item in tokenized]
x_encoded = [item[1] for item in tokenized]
xc_pos = [item[2] for item in tokenized]
xc_width = [item[3] for item in tokenized]
yc_pos = [[y_pos[i]]*len(xc_pos[i]) for i in range(len(texts))]

x_tokens = [subitem for item in x_tokens for subitem in item]
x_encoded = [subitem for item in x_encoded for subitem in item]
xc_pos = [subitem for item in xc_pos for subitem in item]
xc_width = [subitem for item in xc_width for subitem in item]
yc_pos = [subitem for item in yc_pos for subitem in item]
bl = 5 #bigram length

fig_dir = os.path.join(base_dir,'figures')
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

for i in range(20):
    rects = []
    text_objs = []
    colors = ['green']*bl+['red']
    text_params = {'ha': 'center', 'va': 'center', 'fontsize': 10, "fontfamily": 'monospace', "weight": 'bold'}

    # avoid shifting for the lower rectangles. also, keep in mind that when the line changes, the x position will reset
    # So, do not take the xc_pos as it is, and do not use xc_pos[i:i+bl+1]-xc_pos[i]
    lower_x_pos = [0]
    for k in range(bl):
        lower_x_pos.append(xc_width[i+k]+lower_x_pos[k])
    for x_p, y_p, x_w, x_token, x_enc, l_x_p, color in zip(xc_pos[i:i+bl+1], yc_pos[i:i+bl+1], xc_width[i:i+bl+1],
                                                    x_tokens[i:i+bl+1], x_encoded[i:i+bl+1],lower_x_pos,colors):
        rects.append(patches.Rectangle((x_p, y_p), x_w, y_height, facecolor=color, alpha = 0.5))

        if color == colors[-1]:
            x_p = 3
            l_x_p = 3
        rects.append(patches.Rectangle((l_x_p+0.75, -3), x_w, y_height, facecolor=color, alpha = 0.5))

        text_obj1 = ax.text(l_x_p + 0.75 + x_w / 2, -3 + y_height / 2, x_token, **text_params)
        text_obj2 = ax.text(l_x_p + 0.75 + x_w / 2, -3.75 , x_enc,**text_params, color ='grey')

        text_objs.append(text_obj1)
        text_objs.append(text_obj2)

    text_objs.append(ax.text(3.65, -3 + y_height / 2, "→", fontsize = 18, ha = 'right', va = 'center'))
    text_objs.append(ax.text(3.65, -3.75, "→", fontsize = 18, color = 'grey', ha = 'right', va = 'center'))

    for rect in rects:
        ax.add_patch(rect)

    fig.savefig(os.path.join(fig_dir,f'iter {i}.png'), dpi = 300)
    for text in text_objs:
        text.remove()

    for rect in rects:
        rect.remove()


def create_gif_from_list(image_paths, output_gif, duration=500, loop=0, resize=None):
    images = []

    for image_path in image_paths:
        img = Image.open(image_path)

        # Resize if specified
        if resize:
            img = img.resize(resize, Image.Resampling.LANCZOS)

        images.append(img)

    # Save as GIF
    if images:
        images[0].save(
            output_gif,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=loop,
            optimize=True
        )
        print(f"GIF created: {output_gif} with {len(images)} frames")


# Usage

image_list = [os.path.join(fig_dir,f'iter {i}.png') for i in range(20)]
image_list = [image_list[0]]*2 + image_list + [image_list[-1]]*2
output_gif = os.path.join(fig_dir,f'animation.gif')
create_gif_from_list(image_list, output_gif, duration=750, resize=None)

