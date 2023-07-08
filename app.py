#!/usr/bin/env python

import os
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


DESCRIPTION = '# [MaskCut](https://github.com/facebookresearch/CutLER)'

paths = sorted(pathlib.Path('CutLER/maskcut/imgs').glob('*.jpg'))

with gr.Blocks(css='style.css') as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Row():
        with gr.Column():
            image = gr.Image(label='Input image', type='filepath')
            threshold = gr.Slider(
                label='Threshold used for producing binary graph',
                minimum=0,
                maximum=1,
                step=0.01,
                value=0.15)
            max_masks = gr.Slider(
                label='The maximum number of pseudo-masks per image',
                minimum=1,
                maximum=20,
                step=1,
                value=6)
            run_button = gr.Button('Run')
        with gr.Column():
            result = gr.Image(label='Result')

    inputs = [image, threshold, max_masks]
    gr.Examples(examples=[[path.as_posix(), 0.15, 6] for path in paths],
                inputs=inputs,
                outputs=result,
                fn=run,
                cache_examples=os.getenv('CACHE_EXAMPLES') == '1')

    run_button.click(fn=run, inputs=inputs, outputs=result, api_name='run')
demo.queue(max_size=20).launch()
