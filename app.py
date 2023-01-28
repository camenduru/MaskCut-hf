#!/usr/bin/env python

import pathlib

import gradio as gr
import numpy as np
import PIL.Image as Image

from model import Model, random_color, vis_mask

model = Model()


def run(image_path, threshold, max_num_mask):
    image = np.asarray(Image.open(image_path).convert('RGB'))
    masks = model(image_path, threshold, max_num_mask)
    for mask in masks:
        image = vis_mask(image, mask, random_color(rgb=True))
    return image


TITLE = 'MaskCut'
DESCRIPTION = 'This is an unofficial demo for https://github.com/facebookresearch/CutLER.'

paths = sorted(pathlib.Path('CutLER/maskcut/imgs').glob('*.jpg'))
demo = gr.Interface(
    fn=run,
    inputs=[
        gr.Image(label='Input image', type='filepath'),
        gr.Slider(label='Threshold used for producing binary graph',
                  minimum=0,
                  maximum=1,
                  value=0.15,
                  step=0.01),
        gr.Slider(label='The maximum number of pseudo-masks per image',
                  minimum=1,
                  maximum=20,
                  value=3,
                  step=1),
    ],
    outputs=gr.Image(label='Result', type='numpy'),
    examples=[[path.as_posix(), 0.15, 3] for path in paths],
    title=TITLE,
    description=DESCRIPTION)
demo.queue().launch()
