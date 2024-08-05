# Plant Leaf Classification with VGGNet 🌿

## Overview

This project focuses on classifying images of plant leaves into one of 10 categories using a deep learning model based on the VGGNet architecture. The model has been trained on a dataset of potato leaf images, but it can be adapted for other types of plant leaves as well.

## Installation

### Prerequisites

Ensure you have the following libraries installed:

- Python 3.x
- OpenCV (`cv2`)
- NumPy (`numpy`)
- PIL (Pillow)
- Matplotlib (`matplotlib`)
- Keras (`keras`)
- TensorFlow (`tensorflow`)

You can install the dependencies using pip:

```bash
pip install opencv-python numpy pillow matplotlib keras tensorflow
```

## Project Structure

The dataset should be organized in the following directory structure:

```
/path_to_data/
    ├── Class_1/
    │   ├── image_1.jpg
    │   ├── image_2.jpg
    │   └── ...
    ├── Class_2/
    │   ├── image_1.jpg
    │   └── ...
    └── Class_10/
        ├── image_1.jpg
        └── ...
```

## Usage

### 1. Data Loading and Preprocessing

The images are loaded from the specified path, resized to 224x224 pixels, and normalized to prepare for model training.

```python
[X_train, y_train] = read_images("/path_to_data")
```

### 2. Visualizing Images

You can visualize a subset of images from the training dataset using the following function:

```python
plot_images(X_train, total_images=2, rows=1, cols=2, fsize=(10, 50), title='Training Dataset')
```

### 3. Training the Model

The VGGNet model is built and trained on the processed images. Here are the key hyperparameters:

- Image Size: 224x224
- Number of Classes: 10
- Number of Epochs: 1
- Batch Size: 64
- Optimizer: Adam

```python
model = VGGNet.build(input_shape=(224, 224, 3), classes=10)
model.compile(optimizer=optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_features, train_targets, batch_size=64, epochs=1, verbose=1)
```

### 4. Saving the Model

After training, the model is saved as `VGGNet_16_groupe_2.h5`.

```python
model.save("VGGNet_16_groupe_2.h5")
```

### 5. Model Performance

The VGGNet16 model has been evaluated across ten test runs with an average accuracy of **85.25%**. Here are the individual accuracy scores:

1. 86.02%
2. 86.26%
3. 85.95%
4. 86.88%
5. 84.88%
6. 84.42%
7. 83.26%
8. 84.42%
9. 84.79%
10. 83.49%

## Author

This project was created by ISSA IBRAHIM Moubarak.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
