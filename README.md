# BRAIN_SEGMENTATION_USING_CNN
Brain tumor segmentation using U-Net, trained on MRI scans with corresponding masks. The model achieved a high validation score, ensuring accurate tumor detection. Preprocessing includes image normalization, resizing, and augmentation. Implemented in TensorFlow/PyTorch, with training, evaluation, and inference scripts
# DATASETS AND PREPROCESSING
https://figshare.com/articles/dataset/brain_tumor_dataset/1512427
Used all 4 Datasets for training and validation
Contains all .mat files
for smooth workflow need to convert to .npy files
After converting we will have images.npy,labels.npy and masks.npy
for conversion use conversion.py file
we will output shape(766*256*256) shape means 766 images with 256*256 size
As we have 4 folders we need to concatenate all to single file
Combined Data: Images Shape: (3064, 256, 256), Masks Shape: (3064, 256, 256)
#TRAINING
The U-Net model consists of an encoder-decoder architecture for image segmentation. The encoder extracts features using convolutional layers and max-pooling, progressively reducing spatial dimensions. The bottleneck contains high-level features. The decoder uses transposed convolutions and skip connections from the encoder to restore spatial information, producing a pixel-wise segmented output with a sigmoid activation function.
