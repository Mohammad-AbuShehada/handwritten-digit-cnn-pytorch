# Handwritten Digit Recognition

A Python and PyTorch project for training and running a CNN model that recognizes handwritten digits from 0 to 9 using the MNIST dataset. The project also includes a Streamlit interface where users can draw a digit or upload an image and get a prediction.

## Live Demo

[Try the app on Hugging Face Spaces](https://huggingface.co/spaces/mohammad147/handwritten-digit-cnn)



## Features

- CNN-based digit classifier trained on MNIST.
- Saved trained model included as `mnist_model.pth`.
- Predict digits from `PNG`, `JPG`, or `JPEG` images.
- Automatic image preprocessing before prediction: grayscale conversion, background inversion when needed, stroke thickening, centering, resizing to 28x28, and normalization.
- Streamlit app for drawing a digit on a canvas or uploading an image.
- Debug option to save the exact preprocessed image sent to the model.

## Project Structure

```text
.
|-- app.py                 # Streamlit web app
|-- mnist_classifier.py    # MNIST training script
|-- model.py               # DigitCNN model definition
|-- predict_digit.py       # Image prediction script
|-- mnist_model.pth        # Trained model checkpoint
|-- requirements.txt       # Project dependencies
`-- README.md
```

## Requirements

- Python 3.10 or newer
- pip

## Download and Installation

1. Clone the repository:

```powershell
git clone <repository-url>
cd AI
```

If you already have the project folder, open it directly.

2. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Install the dependencies:

```powershell
pip install -r requirements.txt
```

## Run the Streamlit App

Start the web app:

```powershell
streamlit run app.py
```

After running the command, open the URL shown in the terminal. It is usually:

```text
http://localhost:8501
```

In the app, draw a digit or upload an image, then click `Predict` to see the predicted digit and confidence score.

## Predict From an Image

Use an image file that contains one clear handwritten digit:

```powershell
python .\predict_digit.py .\image.png
```

You can also specify the model path manually:

```powershell
python .\predict_digit.py .\image.png --model-path .\mnist_model.pth
```

Example output:

```text
Model loaded successfully. Test accuracy: 99.35%

Predicted Digit: 7
Confidence: 98.29%

Top 3 Predictions:
   7 -> 98.29%
   2 -> 0.75%
   1 -> 0.64%
```

## Train the Model

To train the CNN model on MNIST:

```powershell
python .\mnist_classifier.py --epochs 5
```

The training script downloads MNIST automatically into the `data/` folder and saves the best model checkpoint to:

```text
mnist_model.pth
```

Useful training options:

```powershell
python .\mnist_classifier.py --epochs 10 --batch-size 64 --lr 0.001
python .\mnist_classifier.py --data-dir .\data --model-path .\mnist_model.pth
```

## Debug Preprocessing

To save a preview of the preprocessed 28x28 image that is sent to the model:

```powershell
python .\predict_digit.py .\image.png --debug debug_preprocessed.png
```

To disable stroke thickening during preprocessing:

```powershell
python .\predict_digit.py .\image.png --no-thicken
```

## Image Tips

- Use an image with one digit only.
- Keep the digit centered.
- Avoid extra lines, marks, or noise.
- Use clear contrast between the digit and the background.
- Images with a white background and black digit work well. The script automatically inverts colors when needed.

## Notes

- `mnist_model.pth` is included so prediction can run without retraining.
- The `data/` folder is ignored because MNIST is downloaded automatically during training.
- If PyTorch installation fails, install the correct PyTorch version for your device from the official PyTorch website, then run:

```powershell
pip install -r requirements.txt
```
