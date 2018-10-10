#predict.py


import json
import torch
import predict_args
import warnings

from PIL import Image
from torchvision import models
from torchvision import transforms


def main():
   
    parser = predict_args.get_args()
    parser.add_argument('--version',
                        action='version',
                        version='%(prog)s ' + __version__ + ' by ' + __author__)
    cli_args = parser.parse_args()

    # CPU
    device = torch.device("cpu")

    # GPU
    if cli_args.use_gpu:
        device = torch.device("cuda:0")

    with open(cli_args.categories_json, 'r') as f:
        cat_to_name = json.load(f)

    # model
    my_model = load_checkpoint(device, cli_args.checkpoint_file)

    top_prob, top_classes = predict(cli_args.path_to_image, my_model, cli_args.top_k)

    label = top_classes[0]
    prob = top_prob[0]


    for i in range(len(top_prob)):
        print(f"{cat_to_name[top_classes[i]]:<25} {top_prob[i]*100:.2f}%")


def predict(image_path, model, topk=5):
# model evalution and cpu
    model.eval()
    model.cpu()

    image = process_image(image_path)
    image = image.unsqueeze(0)
    with torch.no_grad():
        output = model.forward(image)
        top_prob, top_labels = torch.topk(output, topk)

        top_prob = top_prob.exp()

    class_to_idx_inv = {model.class_to_idx[k]: k for k in model.class_to_idx}
    mapped_classes = list()

    for label in top_labels.numpy()[0]:
        mapped_classes.append(class_to_idx_inv[label])

    return top_prob.numpy()[0], mapped_classes

#load_checkpoint
def load_checkpoint(device, file='checkpoint.pth'):
  
    model_state = torch.load(file, map_location=lambda storage, loc: storage)

    model = models.__dict__[model_state['arch']](pretrained=True)
    model = model.to(device)

    model.classifier = model_state['classifier']
    model.load_state_dict(model_state['state_dict'])
    model.class_to_idx = model_state['class_to_idx']

    return model

#process_image
def process_image(image):
   
    expects_means = [0.485, 0.456, 0.406]
    expects_std = [0.229, 0.224, 0.225]

    pil_image = Image.open(image).convert("RGB")

    in_transforms = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(expects_means, expects_std)])
    pil_image = in_transforms(pil_image)

    return pil_image


if __name__ == '__main__':
main()
