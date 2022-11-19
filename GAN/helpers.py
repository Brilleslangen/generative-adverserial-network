from matplotlib import pyplot as plt
import torchvision.utils as tvutils
import numpy as np
import os


# Either displays or saves 32 images
# If directory or filename is None, images will be displayed, but not saved.
# If directory and filename both are filled, the images will be saved to the given directory with the given
# filename, but the images will not be displayed.
def display_images(images, directory=None, filename=None):
    fig = plt.figure(figsize=(12, 12))
    plt.axis("off")
    plt.title(f'{filename}')
    plt.imshow(np.transpose(tvutils.make_grid(images[0][:32], padding=2, normalize=True).cpu(), (1, 2, 0)))
    if filename is None or directory is None:
        plt.show()
    else:
        path = f'../results/{directory}'
        initiate_directory('/results')
        initiate_directory(path)
        plt.savefig(f'{path}/{filename}.png', bbox_inches='tight')
    plt.close(fig)


def initiate_directory(path):
    if not os.path.isdir(path):
        os.mkdir(path)
