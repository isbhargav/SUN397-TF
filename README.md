# Classification on SUN397 dataset
# ________________________________________________________

- **Dataset:** The Dataset selected for the Final Project is SUN397 which belongs to Task-
    2. The dataset contains different classes belonging to environmental scenes, places and
    the objects within. The database contains 108,753 images of 397 categories, used in the Scene
    Understanding (SUN) benchmark. The number of images varies across categories, but there
    are at least 100 images per category.
- **Training/Testing Partition:** For the results of this experiment we have chosen at
    random one of the available ten training and testing partitions provided by the authors
    of the dataset. The randomly chosen partition is the fifth partition which has two files
    namely: **Training_05.txt** and **Testing_05.txt**. Each of these files contains
    19850(50x397) image paths where each class has fifty images belonging to it.
- **Image Augmentation:** Each image undergoes different augmentation methods. The
    methods used are listed as follows:
    **1. Rotation:** Degree range for random rotations(-10, +10).
    **2. Symmetric Warp:** Change the angle at which image is viewed.(-0.2, +0.2)
    **3. Zoom range:** Range for random zoom(1, 1.1).
    **4. Left Right flip:** Randomly flip inputs left or right.
    **5. Brightness:** Changes brightness of Images(0.4, 0.6).
    **6. Contrast:** Changes contrast of Images(0.8, 1.25).
    **7. Crop and Pad:** Randomly crops and pads the remaining part of image.
- **Feature Fusion Technique:** We have used adaptive2D pooling method for feature
    fusion. This method combines max pooled and average pooled features and then
    passes it to fully connected layers.
- **Classifier Model:** The classification method used in this experiment is D-CNN. The
    model used in the experiment is ResNet 50. We have trained this network structure by
    using the Place365 weights as the initial weights. The default input size for this model
    is (3, 224,224).
- **Architecture:** The architecture used in this experiment contains the Resnet 50
    structure where the dense layers are replaced with an adaptive2D pooling layer
    followed by fully connected layers. The model architecture for the top layers is as
    shown in figure-1:


- **Training Process:** We have executed the training process in two steps.

```
Step-1: The training occurs for only the fully connected layers for the initial 5 epochs.
This step finetunes our classifier for the given dataset.
Step-2: The convolution layers are unfreezed and all the layers are trained for 2
epochs. This step finetunes our model for the given dataset.
  Parameters for model:
  **1. Optimizer:** Adam
  **2. Loss_Function:** Categorical **_** Crossentropy
 ```
**Discriminative Learning Rate:** Differential Learning Rates (LR) are used for faster,
more efficient transfer learning. The general idea is to use a lower Learning Rate for
the earlier layers, and gradually increase it in the latter layers.

**One-cycle Policy:** Varying the learning rate between reasonable bounds can actually increase
the accuracy of the model in fewer steps. Here, we start with a small learning rate (like 1e-4,
1e-3) and increase the lr after each mini-batch till the loss starts exploding. Once loss starts
exploding stop the range test run. Then plot the learning rate vs loss and choose the learning
rate at-least one order lower than the learning rate where the loss is minimum.

- **Results:** The results obtained after both the training steps for three runs is as shown
    in igure- 2. The average testing accuracy obtained after three runs is **69.313** %.


- **Training time:** Average training time for each run is **57 min 15 s.**
- **Log Files:** The log files for all the runs are available in the project directory.
- **Model Weights:** The trained model weights are available in the models folder.

**NOTE:** To replicate the experiment follow the instructions mentioned in the readme.txt
file.


