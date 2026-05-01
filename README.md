# Handwritten Digit Recognition

A PyTorch project for training and using a CNN model to recognize handwritten digits from image files.

The model is trained on the MNIST dataset and includes preprocessing for real handwritten input, such as white backgrounds, thin strokes, resizing, centering, and normalization.

## 🚀 Live Demo
[Try the app online](https://handwritten-digit-cnn-pytorch-4rzfxq6qhntzelfudakkbi.streamlit.app/)

## Features
- CNN-based digit classifier
- MNIST training script
- Image prediction script
- Automatic preprocessing for handwritten images
- Optional debug output to preview the exact model input
- Saved trained model included as `mnist_model.pth`

## Project Structure
```text
.
|-- model.py
|-- mnist_classifier.py
|-- predict_digit.py
|-- app.py                 # Streamlit web demo
|-- mnist_model.pth
|-- requirements.txt
`-- README.md
Installation
PowerShellpip install -r requirements.txt
Train the Model
PowerShellpython .\mnist_classifier.py --epochs 5
The training script downloads MNIST automatically and saves the best model to:
textmnist_model.pth
Predict a Digit
PowerShellpython .\predict_digit.py .\image.png
Example output:
textModel loaded successfully. Test accuracy: 99.35%

Predicted Digit: 7
Confidence: 98.29%

Top 3 Predictions:
   7 -> 98.29%
   2 -> 0.75%
   1 -> 0.64%
Web Demo (Streamlit)
PowerShellstreamlit run app.py
Debug Preprocessing
Use --debug to save a preview of the 28x28 image that is sent to the model:
PowerShellpython .\predict_digit.py .\image.png --debug debug_image.png
Useful Options
PowerShellpython .\predict_digit.py .\image.png --model-path mnist_model.pth
python .\predict_digit.py .\image.png --no-thicken
Input Tips
For best results, use an image with:

One digit only
A centered digit
No extra lines or marks
Clear contrast between digit and background

Requirements

Python 3.10+
PyTorch
Torchvision
Pillow
