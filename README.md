# Deep Convolutional GAN with Multi-Scale Gradients
This is a project in the course *IDATT2502 - Applied Machine Learning* at the Norwegian University of Science and Technology (NTNU) by:
- Aleksander Brekke Røed
- Erlend Rønning
- Nicolai Thorer Sivesind

In this project we have developed a **Deep Convolutional Generative Adversarial Network with Multi-Scale Gradients**.

## Available datasets
We have added some datasets to the execute file to test and train the GAN.
Currently, we have these datasets available:

| Dataset index | Dataset Content             |
|---------------|-----------------------------|
| 0             | MNIST - Handwritten numbers |
| 1             | Abstract Art Gallery        |
| 2             | Bored Apes Yacht Club       |
| 3             | CelebA                      |

You can choose what dataset you want to train on by passing the selected *dataset index*
to the *run-function* in *execute.py*

### Pre-requisites 
1. **Install Dependencies**
   
    Install the packages in the file `requirements.txt`. 
    You can do this by opening a terminal in this directory and running:

    ```console
    pip install -r requirements.txt
    ```

2. **Retrieve Kaggle API-token**
   
    The application will download the selected dataset for you from kaggle through [opendatasets](https://pypi.org/project/opendatasets/), but Kaggle requires the use of a personal API-token.
   1. Create a Kaggle user at www.kaggle.com
   2. Go to https://kaggle.com/me/account
   3. Scroll down to the section of the page labelled API, and click the *Create New API token*-button.
      
      This will download a file called "kaggle.json" which contains your username and an API-key. 
      ```json
      {"username":"YOUR_KAGGLE_USERNAME", "key":"YOUR_KAGGLE_KEY"}
      ```
      Store it somewhere you can easily access it. You will need to fill in both username and API-token in the terminal later.
   
### How to use
If you have completed both steps in [*pre-requisites*](#pre-requisites) you are now ready to train and run the GAN.

1. Open this directory in your choice of IDE. 
2. Run the *execute.py* file.
   This will run some pre-defined training configurations of each dataset.

#### Running you own configuration
It's pretty simple to modify and run a training configuration:
1. Scroll to the bottom of the *execute.py* file.
2. Remove any calls to the *run*-function.
3. Add your own call to the *run*-function with selected parameters. Descriptions of each parameter follow.

##### Parameters of *run()*
**ds_index**: 
Dataset index. This is the index of the dataset you want to train the GAN on. 
You can look up the indexes [here](#available-datasets). 

**epochs**: Number of Training rounds before completion. The length of an epoch depends on the dataset. 
A single epoch is completes when all instances in the dataset has been trained on.

**image_size**: Amount of pixels of the sides of quadratic images. For example will image_size=64 make training and output images
have shape (64, 64, c).

**conv_scalar**: Convolution scalar. This value decides the number of channels in the convolutional layers, which have to be lowered
from the image size when working with resolutions higher than 64x64. Note: Has to be a power of two for the scaling algorithm to work.

**num_conv_layers**: Number of convolutional layers in the discriminator and convolutional transposed layers in the generator, in addition to the first and last layers. Recommended values are 3 layers for 64x64 pixel images, and 4 layers for 128x128 pixel images.

**lr**: Learning rate for both networks. The value scales the magnitude of the network's weight updates to minimize the loss function. Redford et al. recommends this value being 0.002 in the DCGAN paper, but with MSG enabled the value needs to be lowered for some datasets.

**model_name**: Name of the model for efficient logging and differentiating outputs from different approaches.

**preview_images**: Boolean value deciding if a preview of the chosen dataset is shown before training.

**display_frequency**: How often benchmark images should be saved. All images will be saved to *results/<ds_name>/*.
A frequency of *1* will make the GAN save a grid of 32 generated images after each epoch completion. The saved image grid
will be generated from the same benchmark noise every epoch such that the evolution of the generator can be tracked.

**tf_model**: Path to model to transfer learn. The model at the given path will be loaded into the GAN, and continue training as normal. If None is given, the model
is created from scratch.

**msg**: Boolean choosing wether to enable multi-scale gradients in the GAN.  

##### Example of a configuration
Heres an example of running the Gan on the *Abstract Art* dataset with *200 epochs* and a *display_frequency* of 10, resulting
in 20 saved image grids.

```python
run(ds_index=1, epochs=200, image_size=64, conv_scalar=64, num_conv_layers=3, msg=False)
```

##### Displaying images from a pre-trained model
Output from a model can be displayed by calling the *display_images_from_model()*-function. It takes the index of the dataset, model name and image size as its parameters.
