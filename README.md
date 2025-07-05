# Simple_ai
simple beginner pytorch project, CNN that classifies objects in a 32X32 pixel image trained on CIFAR10 dataset
training model --> simpleai.py (this trains a model and saves it in the program folder and tests it using an image in the program folder)
accessing saved model --> testing.py (this loads an uses the saved model using an image in the program folder)

images do not need to be resized ;)

15 epochs - 555.96 seconds (ryzen 7 4700u 16gb DDR4, no dGPU)
            367.71 seconds (ryzen ai 9 365, 24gb LPDDR5x, no dgpu
            191.65 seconds ([Google colab](colab.research.google.com), hardware accelerator: T4-GPU)
