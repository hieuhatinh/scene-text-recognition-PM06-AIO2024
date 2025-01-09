import torch
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from models.crnn import CRNN
from PIL import Image
import io
from torchvision import transforms
import numpy as np


class OCRService:
    def __init__(self, detect_model, reg_model, idx_to_char) -> None:
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.detect = detect_model.to(self.device)
        self.recognition = reg_model.to(self.device)
        self.transform = transforms.Compose(
            [
                transforms.Resize((100, 420)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ]
        )
        self.idx_to_char = idx_to_char

    def decode(self, encoded_sequences, idx_to_char, blank_char='-'):
        decoded_sequences = []
        for seq in encoded_sequences:
            decoded_label = []
            prev_char = None

            for token in seq:
                if token != 0:
                    char = idx_to_char[token.item()]
                    if char != blank_char:
                        if char != prev_char or prev_char == blank_char:
                            decoded_label.append(char)
                    prev_char = char
            decoded_sequences.append(''.join(decoded_label))
        print(f'From {encoded_sequences} to {decoded_sequences}')
        return decoded_sequences

    def text_detection(self, image):
        text_det_results = self.detect(image, verbose=False)[0]
        bboxes = text_det_results.boxes.xyxy.tolist()
        classes = text_det_results.boxes.cls.tolist()
        names = text_det_results.names
        confs = text_det_results.boxes.conf.tolist()
        return bboxes, classes, names, confs

    def text_recognition(self, img, data_transforms, text_reg_model, idx_to_char, device):
        transformed_image = data_transforms(img)
        transformed_image = transformed_image.unsqueeze(0).to(device)
        text_reg_model.eval()
        with torch.no_grad():
            logits = text_reg_model(transformed_image).detach().cpu()
        text = self.decode(logits.permute(1, 0, 2).argmax(2), idx_to_char)
        return text

    async def draw_prediction(self, image, predictions):
        img_array = np.array(image)

        annotator_img = Annotator(
            img_array,
            font='Arial.ttf',
            pil=False
        )

        # Sort predictions by y-coordinate to handle overlapping better
        predictions = sorted(
            predictions, key=lambda x: x[0][1]
        )  # Sort by y1 coordinate

        for bbx, text, cls_name, conf in predictions:
            x1, y1, x2, y2 = bbx
            label = f'{cls_name} {conf:.1f}:{text}'

            # background color
            background_color = colors(hash(cls_name) % 20, True)
            # get text color
            r, g, b = background_color
            brightness = (r * 299 + g * 587 + b * 114) / 1000
            text_color = (255, 255, 255) if brightness < 128 else (0, 0, 0)

            background_color_bgr = (b, g, r)

            annotator_img.box_label(
                (x1, y1, x2, y2),
                label,
                color=background_color_bgr,
                txt_color=(text_color)
            )
        return Image.fromarray(annotator_img.result())

    async def process_image(self, file_image):
        image = Image.open(io.BytesIO(file_image))
        bboxes, classes, names, confs = self.text_detection(image)
        predict_texts = []
        for bbx, cls, conf in zip(bboxes, classes, confs):
            x1, y1, x2, y2 = bbx
            img_crop = image.crop((x1, y1, x2, y2))
            name = names[int(cls)]
            text = self.text_recognition(
                img_crop, self.transform, self.recognition, self.idx_to_char, self.device)
            predict_texts.append((bbx, text, name, conf))
        return predict_texts


# model path
crnn_model_path = 'E:/AIO_document/AIO2024/AIO2024-Project/scene-text-recognition-PM06-AIO2024/src/weights/crnn.pt'
yolo_model_path = 'E:/AIO_document/AIO2024/AIO2024-Project/scene-text-recognition-PM06-AIO2024/src/weights/best.pt'

# text detection model
yolo = YOLO(yolo_model_path)

# text recognition model
chars = '0123456789abcdefghijklmnopqrstuvwxyz-'
vocab_size = len(chars)
char_to_idx = {char: idx+1 for idx, char in enumerate(sorted(chars))}
idx_to_char = {index: char for char, index in char_to_idx.items()}

hidden_size = 256
n_layers = 3
dropout_prob = 0.2
unfreeze_layers = 3
device = "cuda" if torch.cuda.is_available() else "cpu"

crnn_model = CRNN(
    vocab_size=vocab_size,
    hidden_size=hidden_size,
    n_layers=n_layers,
    dropout=dropout_prob,
    unfreeze_layers=unfreeze_layers,
).to(device)
crnn_model.load_state_dict(torch.load(crnn_model_path))

ocr_service = OCRService(yolo, crnn_model, idx_to_char)
