import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image, ImageChops, ImageFilter, ImageOps, ImageStat

from model import DigitCNN


def has_light_background(image):
    width, height = image.size
    edge = max(2, min(width, height) // 20)
    regions = [
        image.crop((0, 0, width, edge)),
        image.crop((0, height - edge, width, height)),
        image.crop((0, 0, edge, height)),
        image.crop((width - edge, 0, width, height)),
    ]
    return sum(ImageStat.Stat(region).mean[0] for region in regions) / len(regions) > 127


def center_digit(image, canvas_size=28, digit_size=20):
    mask = image.point(lambda pixel: 255 if pixel > 20 else 0)
    bbox = mask.getbbox()
    if bbox:
        image = image.crop(bbox)

    image.thumbnail((digit_size, digit_size), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (canvas_size, canvas_size), 0)
    x = (canvas_size - image.width) // 2
    y = (canvas_size - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def preprocess_image(image_path, debug_path=None, thicken=True):
    image = Image.open(image_path).convert("L")
    image = ImageOps.autocontrast(image)

    if has_light_background(image):
        image = ImageOps.invert(image)

    image = ImageChops.multiply(image, image)
    if thicken:
        image = image.filter(ImageFilter.MaxFilter(3))

    image = center_digit(image)

    if debug_path:
        image.resize((280, 280), Image.Resampling.NEAREST).save(debug_path)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    return transform(image).unsqueeze(0)


def load_model(model_path):
    model = DigitCNN()
    checkpoint = torch.load(model_path, weights_only=False, map_location="cpu")

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        accuracy = checkpoint.get("accuracy")
    else:
        raise ValueError(
            "This model file uses the old Linear network. "
            "Run: python mnist_classifier.py --epochs 5"
        )

    model.eval()
    return model, accuracy


def predict(model, input_tensor):
    with torch.no_grad():
        probabilities = F.softmax(model(input_tensor)[0], dim=0)
        top3 = torch.topk(probabilities, 3)

    return [
        (top3.indices[i].item(), top3.values[i].item() * 100)
        for i in range(3)
    ]


def main():
    parser = argparse.ArgumentParser(description="Predict a handwritten digit image.")
    parser.add_argument("image", help="Path to a PNG/JPG digit image")
    parser.add_argument("--model-path", default="mnist_model.pth")
    parser.add_argument("--debug", nargs="?", const="debug_preprocessed.png")
    parser.add_argument(
        "--no-thicken",
        action="store_true",
        help="Disable stroke thickening during preprocessing",
    )
    args = parser.parse_args()

    try:
        model, accuracy = load_model(args.model_path)
        if accuracy is not None:
            print(f"Model loaded successfully. Test accuracy: {accuracy:.2f}%")
        else:
            print("Model loaded successfully.")

        input_tensor = preprocess_image(
            args.image,
            debug_path=args.debug,
            thicken=not args.no_thicken,
        )
        if args.debug:
            print(f"Saved model input preview: {Path(args.debug)}")

        top3 = predict(model, input_tensor)
        predicted_digit, confidence = top3[0]

        print(f"\nPredicted Digit: {predicted_digit}")
        print(f"Confidence: {confidence:.2f}%")
        print("\nTop 3 Predictions:")
        for digit, conf in top3:
            print(f"   {digit} -> {conf:.2f}%")

    except Exception as error:
        print(f"Error: {error}")


if __name__ == "__main__":
    main()
