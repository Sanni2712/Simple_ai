# Simple_ai
Simple beginner pytorch project, CNN that classifies objects in a 32X32 pixel image trained on CIFAR10 dataset
training model --> `simpleai.py` <br>
(this trains a model and saves it in the program folder and tests it using an image in the program folder)<br><br>
accessing saved model --> `testing.py` (this loads an uses the saved model using an image in the program folder)

images do not need to be resized ;)

15 epochs -<br> 555.96 seconds (ryzen 7 4700u 16gb DDR4, no dGPU)<br>
                367.71 seconds (ryzen ai 9 365, 24gb LPDDR5x, no dgpu)<br>
                191.65 seconds ([Google colab](colab.research.google.com), hardware accelerator: T4-GPU)<br><br>
## Requirements:
[Google colab](colab.research.google.com) - no installaton needed
<br><br>
local - 
<br>torch-
<br>`pip3 install torch`<br>
<br>torchvision -
<br>`pip3 install torchvision`
<br>
<br>Pillow
<br>`pip3 install Pillow`


