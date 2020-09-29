# Image Animation

## Installation

It support `python3`. To install the dependencies run:
```
pip install -r requirements.txt
```

For cropping videos during animation process, you need to install `ffmpeg` and `face-alligment`:
```bash
conda install ffmpeg
pip install face-alligment
```

## Pre-trained model

Download `vox-adv-cpk.pth.tar` from [here](https://drive.google.com/drive/folders/1PyQJmkdCsAkOYwUyaj_l-l0as-iLDgeH) and put it in `./`.


## How to use?

### Import images
```python
from PIL import Image
img = Image.open(PATH)
img_ls = [img1, img2, ...]
```

### Face detection
```python
from image_animation import FaceDetect
detector = FaceDetect()
boxes, probs, annotates, faces = detector.detect(img_ls, crop_size=None, mode = 'Extract_largest')
```
If images in list are not in a same size, set `crop_size`.
There're 4 modes: `Detect_bool`, `Detect`, `Extract_largest`, `Extract_all`.
Images of faces will be saved in `face_result/faces`.
If you want to get the annotation image (face boxes on image), set `save_annotate=True` and it will be save in `face_result/annotations`.

Use `help(FaceDetect)` to see more arguments and details.


### Animation
```python
from image_animation import Animation
animator = Animation()
animator.animate(source_video = SOURCE_VIDEO, source_image = 'image_animation/image/einstein.png', result_video = 'result.mp4')
```
`source_video` is the video from camera.
`source_image` is the image path of the physicist you want to become, defaults to `image_animation/image/einstein.png`.
And the result will be saved into `result_video`, defaults to `result.mp4`.


## Inference time
The run time is about 29.14s with 1 GPU(GTX 1080).