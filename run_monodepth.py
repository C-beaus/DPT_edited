"""Compute depth maps for images in the input folder.
"""
import os
import glob
import torch
import cv2
import argparse

import util.io

from torchvision.transforms import Compose
import torchvision.transforms as transforms

from dpt.models import DPTDepthModel
from dpt.midas_net import MidasNet_large
from dpt.transforms import Resize, NormalizeImage, PrepareForNet

from torch.utils.data import Dataset, DataLoader

from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import logging
import sys

#from util.misc import visualize_attention

# Set up logging
logger = logging.getLogger('train_logger')
logger.setLevel(logging.DEBUG)

# Create handlers for console and file logging
c_handler = logging.StreamHandler(sys.stdout)
f_handler = logging.FileHandler('train.log')

c_handler.setLevel(logging.DEBUG)
f_handler.setLevel(logging.DEBUG)

# Create formatters and add them to handlers
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
c_handler.setFormatter(formatter)
f_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(c_handler)
logger.addHandler(f_handler)

class SyndroneDatasetTrain(Dataset):
    def __init__(self, train_image_paths, train_depth_paths, transform, device):
        self.train_image_paths = train_image_paths
        self.train_depth_paths = train_depth_paths
        self.transform = transform
        self.device = device
    def __len__(self):
        return len(self.train_image_paths)
    def __getitem__(self, idx):
        # rgb_image = cv2.imread(self.train_image_paths[idx], cv2.IMREAD_UNCHANGED)[...,::-1]/255. # maybe edit if I have bounding boxes over the image?
        rgb = util.io.read_image(self.train_image_paths[idx])
        depth = util.io.read_image(self.train_depth_paths[idx])

        rgb_image = self.transform({"image": rgb})["image"]
        # depth_image = self.transform({"image" : rgb, "depth": depth})["depth"]
        depth_image = self.transform({"image" : depth})["image"]


        rgb_sample = torch.from_numpy(rgb_image).to(self.device).unsqueeze(0)
        depth_sample = torch.from_numpy(depth_image).to(self.device).unsqueeze(0)

        # depth_image = cv2.imread(self.train_depth_paths[idx]) # Is this correct?

        # if self.transform:
        #     rgb_image = self.transform(rgb_image)
        #     depth_image = self.transform(depth_image)
        return {"rgb": rgb_sample, "depth": depth_sample}
    
class SyndroneDatasetVal(Dataset):
    def __init__(self, val_image_paths, val_depth_paths, transform, device):
        self.val_image_paths = val_image_paths
        self.val_depth_paths = val_depth_paths
        self.transform = transform
        self.device = device
    def __len__(self):
        return len(self.val_image_paths)
    def __getitem__(self, idx):
        # rgb_image = cv2.imread(self.val_image_paths[idx], cv2.IMREAD_UNCHANGED)[...,::-1]/255. # maybe edit if I have bounding boxes over the image?
        # depth_image = cv2.imread(self.val_depth_paths[idx]) # Is this correct?

        # if self.transform:
        #     rgb_image = self.transform(rgb_image)
        #     depth_image = self.transform(depth_image)

        rgb = util.io.read_image(self.val_image_paths[idx])
        depth = util.io.read_image(self.val_depth_paths[idx])

        rgb_image = self.transform({"image": rgb})["image"]
        # depth_image = self.transform({"depth": depth})["depth"]
        depth_image = self.transform({"image" : depth})["image"]


        rgb_sample = torch.from_numpy(rgb_image).to(self.device).unsqueeze(0)
        depth_sample = torch.from_numpy(depth_image).to(self.device).unsqueeze(0)
        return {"rgb": rgb_sample, "depth": depth_sample}
    
def custom_loss(predicted_depth, ground_truth_depth):
    mask = (ground_truth_depth > 0).float()
    return torch.mean(torch.abs(predicted_depth - ground_truth_depth) * mask)


def run(input_path, output_path, model_path, train_bool, model_type="dpt_hybrid", optimize=True):
    """Run MonoDepthNN to compute depth maps.

    Args:
        input_path (str): path to input folder
        output_path (str): path to output folder
        model_path (str): path to saved model
    """
    print("initialize")

    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: %s" % device)

    # load network
    if model_type == "dpt_large":  # DPT-Large
        net_w = net_h = 384
        model = DPTDepthModel(
            path=model_path,
            backbone="vitl16_384",
            non_negative=True,
            enable_attention_hooks=False,
        )
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif model_type == "dpt_hybrid":  # DPT-Hybrid
        net_w = net_h = 384
        model = DPTDepthModel(
            path=model_path,
            backbone="vitb_rn50_384",
            non_negative=True,
            enable_attention_hooks=False,
        )
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif model_type == "dpt_hybrid_kitti":
        net_w = 1216
        net_h = 352

        model = DPTDepthModel(
            path=model_path,
            scale=0.00006016,
            shift=0.00579,
            invert=True,
            backbone="vitb_rn50_384",
            non_negative=True,
            enable_attention_hooks=False,
        )

        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif model_type == "dpt_hybrid_nyu":
        net_w = 640
        net_h = 480

        model = DPTDepthModel(
            path=model_path,
            scale=0.000305,
            shift=0.1378,
            invert=True,
            backbone="vitb_rn50_384",
            non_negative=True,
            enable_attention_hooks=False,
        )

        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif model_type == "midas_v21":  # Convolutional model
        net_w = net_h = 384

        model = MidasNet_large(model_path, non_negative=True)
        normalization = NormalizeImage(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    else:
        assert (
            False
        ), f"model_type '{model_type}' not implemented, use: --model_type [dpt_large|dpt_hybrid|dpt_hybrid_kitti|dpt_hybrid_nyu|midas_v21]"

    transform = Compose(
            [
                Resize(
                    net_w,
                    net_h,
                    resize_target=None,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=32,
                    resize_method="minimal",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                normalization,
                PrepareForNet(),
            ]
        )

    if train_bool:
        model.to(device)
        model_params = Params()
        epochs = model_params.num_epochs
        start_epoch = 0
        resume_training = model_params.resume_training

        # Example usage
        train_directory = model_params.train_directory  # Replace with your source folder path
        val_directory = model_params.val_directory  # Replace with your target folder path
        file_extension = ".jpg"  # You can change this to ".png" or other image formats
        depth_file_extension = ".png"

        depth_train_directory = model_params.depth_train_directory
        depth_val_directory = model_params.depth_val_directory
        
        train_image_paths = []
        val_image_paths = []
        depth_train_paths = []
        depth_val_paths = []

        # RGB images
        # Loop through all files in the train directory
        for filename in os.listdir(train_directory):
            # Check if the file has the specified image extension
            if filename.lower().endswith(file_extension):
                # Construct full file paths
                img_path = os.path.join(train_directory, filename)
                train_image_paths.append(img_path)
        # Loop through all files in the val directory
        for filename in os.listdir(val_directory):
            if filename.lower().endswith(file_extension):
                img_path = os.path.join(val_directory, filename)
                val_image_paths.append(img_path)

        # Depth images
        for filename in os.listdir(depth_train_directory):
            # Check if the file has the specified image extension
            if filename.lower().endswith(depth_file_extension):
                # Construct full file paths
                img_path = os.path.join(depth_train_directory, filename)
                depth_train_paths.append(img_path)
        # Loop through all files in the val directory
        for filename in os.listdir(depth_val_directory):
            if filename.lower().endswith(depth_file_extension):
                img_path = os.path.join(depth_val_directory, filename)
                depth_val_paths.append(img_path)
                

        train_dataset = SyndroneDatasetTrain(train_image_paths, depth_train_paths, transform, device)
        val_dataset = SyndroneDatasetVal(val_image_paths, depth_val_paths, transform, device)
        train_dataloader = DataLoader(train_dataset, batch_size=model_params.batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=True) # val considers entire val dataset

        # Construct an optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=model_params.lr)

        checkpoint_path = os.path.join("checkpoints", model_params.run_title, f"checkpoint.pth")

        if resume_training and os.path.exists(checkpoint_path):
            print("reloading model from last checkpoint")
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint["model"])
            start_epoch = checkpoint["epoch"] + 1
            optimizer.load_state_dict(checkpoint["optimizer"])
            # lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            assert model_params == checkpoint["model_params"]

        Path(os.path.join("runs/", f"{model_params.run_title}/", 'train')).mkdir(parents=True, exist_ok=True)
        Path(os.path.join("checkpoints", model_params.run_title)).mkdir(parents=True, exist_ok=True)

        print(f"start epoch: {start_epoch}")
        print("Training")
        for epoch in range(start_epoch, epochs):
            model.train()
            epoch_loss = 0.0

            for batch in train_dataloader:
                rgb_image = batch["rgb"].to(device)
                ground_truth_depth = batch["depth"].to(device)
                
                # loss_temp = []
                loss_temp = 0
                for rgb, ground_truth in zip(rgb_image, ground_truth_depth):

                    # predicted_depth = model.forward(rgb_image)
                    # loss = (custom_loss(predicted_depth, ground_truth_depth))
                    predicted_depth = model.forward(rgb)

                    # loss_temp.append(custom_loss(predicted_depth, ground_truth))
                    loss_temp += custom_loss(predicted_depth, ground_truth)

                # loss = sum(loss_temp)
                loss = loss_temp
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            #     torch.cuda.empty_cache() # Not entirely sure if this will help or not

            # scaler = torch.cuda.amp.GradScaler()  # Initialize scaler
            #
            # for batch in train_dataloader:
            #     with torch.cuda.amp.autocast():  # Enable mixed precision
            #         rgb_image = batch["rgb"]
            #         ground_truth_depth = batch["depth"]
            #         loss_temp = 0
            #         for rgb, ground_truth in zip(rgb_image, ground_truth_depth):

            #             # predicted_depth = model.forward(rgb_image)
            #             # loss = (custom_loss(predicted_depth, ground_truth_depth))
            #             rgb = rgb.to(device)
            #             ground_truth = ground_truth.to(device)

            #             predicted_depth = model.forward(rgb)

            #             # loss_temp.append(custom_loss(predicted_depth, ground_truth))
            #             loss_temp += custom_loss(predicted_depth, ground_truth)

            #             # predicted_depth = model.forward(rgb_image)
            #             # loss = custom_loss(predicted_depth, ground_truth_depth)
            #     loss = loss_temp
            #     scaler.scale(loss).backward()  # Scale loss
            #     optimizer.zero_grad()
            #     scaler.step(optimizer)
            #     scaler.update()
            #     torch.cuda.empty_cache() # Not entirely sure if this will help or not

                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(train_dataloader)
            print(f"Epoch [{epoch + 1}/{epochs}], Average Loss: {avg_loss:.4f}")
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                # "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                # "model_params": model_params
            }
            
            torch.save(checkpoint, os.path.join("checkpoints", model_params.run_title, f"model_{epoch}.pth"))
            torch.save(checkpoint, os.path.join("checkpoints", model_params.run_title, f"checkpoint.pth"))
            writer_train = SummaryWriter(f'runs/{model_params.run_title}/train')
            writer_train.add_scalar('average_training_loss', avg_loss, epoch)
        
        # Save the trained model
        Path(os.path.join("models", model_params.run_title)).mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), os.path.join('models', model_params.run_title, 'trained_model.pt'))

        # Define the file path where parameters will be saved
        file_path = os.path.join("models", f"{model_params.run_title}", "parameters.txt")
        parameters = model_params.get_params()

        # Write the parameters to the text file
        with open(file_path, 'x') as f:
            for param, value in parameters.items():
                # Write the parameter name and value in the format "param_name = value"
                f.write(f"{param} = {value}\n")
        logger.info("Training completed, params saved, and model saved to 'fasterrcnn_model_" + model_params.run_title + ".pt'.")

    else:
        print("Evaluating, No Training")
        model.eval()

        if optimize == True and device == torch.device("cuda"):
            model = model.to(memory_format=torch.channels_last)
            model = model.half()

        model.to(device)

        # get input
        img_names = glob.glob(os.path.join(input_path, "*"))
        num_images = len(img_names)

        # create output folder
        os.makedirs(output_path, exist_ok=True)

        print("start processing")
        for ind, img_name in enumerate(img_names):
            if os.path.isdir(img_name):
                continue

            print("  processing {} ({}/{})".format(img_name, ind + 1, num_images))
            # input

            img = util.io.read_image(img_name)

            if args.kitti_crop is True:
                height, width, _ = img.shape
                top = height - 352
                left = (width - 1216) // 2
                img = img[top : top + 352, left : left + 1216, :]

            img_input = transform({"image": img})["image"]

            # compute
            with torch.no_grad():
                sample = torch.from_numpy(img_input).to(device).unsqueeze(0)

                if optimize == True and device == torch.device("cuda"):
                    sample = sample.to(memory_format=torch.channels_last)
                    sample = sample.half()

                prediction = model.forward(sample)
                prediction = (
                    torch.nn.functional.interpolate(
                        prediction.unsqueeze(1),
                        size=img.shape[:2],
                        mode="bicubic",
                        align_corners=False,
                    )
                    .squeeze()
                    .cpu()
                    .numpy()
                )

                if model_type == "dpt_hybrid_kitti":
                    prediction *= 256

                if model_type == "dpt_hybrid_nyu":
                    prediction *= 1000.0

            filename = os.path.join(
                output_path, os.path.splitext(os.path.basename(img_name))[0]
            )
            util.io.write_depth(filename, prediction, bits=2, absolute_depth=args.absolute_depth)

        print("finished")


class Params:
    def __init__(self):
        self.num_epochs = 100
        self.batch_size = 4
        self.lr = 0.005
        # self.momentum = 0.9
        # self.weight_decay = 0.0001 # 0.0005
        # self.lr_step_size = 18 #3
        # self.lr_gamma = 0.1
        # self.name = "resnet_backbone" # mobilenet_backbone #resnet_backbone
        self.resume_training = True
        self.run_title = "syndrone_train_v2"
        # self.train_directory = 'c:/Users/chase/OneDrive/Documents/Grad/ML_for_Robots/final_project/dataset/images/train'
        # self.val_directory = 'c:/Users/chase/OneDrive/Documents/Grad/ML_for_Robots/final_project/dataset/images/val'
        # self.depth_train_directory = 'c:/Users/chase/OneDrive/Documents/Grad/ML_for_Robots/final_project/dataset/depth/train'
        # self.depth_val_directory = 'c:/Users/chase/OneDrive/Documents/Grad/ML_for_Robots/final_project/dataset/depth/val'
        self.train_directory = 'dataset/images/train'
        self.val_directory = 'dataset/images/val'
        self.depth_train_directory = 'dataset/depth/train'
        self.depth_val_directory = 'dataset/depth/val'
        # self.train_bool = True

    def get_params(self):
        parameters = {
            'num_epochs': self.num_epochs,
            'batch_size': self.batch_size,
            'lr': self.lr,
            # 'momentum': self.momentum,
            # 'weight_decay': self.weight_decay,
            # 'lr_step_size': self.lr_step_size,
            # 'lr_gamma': self.lr_gamma,
            # 'name': self.name,
            'resume_training': self.resume_training,
            'run_title': self.run_title,
            'train_directory' : self.train_directory,
            'val_directory' : self.val_directory,
            'depth_train_directory' : self.depth_train_directory,
            'depth_val_directory' : self.depth_val_directory
        }
        return parameters

    def __repr__(self):
        return str(self.__dict__)
    
    def __eq__(self, other):
        return self.__dict__ == other.__dict__


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c", # c for choice
        "--train_bool",
        default=True,
        help="Set to True if you want to train the model"
    )

    parser.add_argument(
        "-i", "--input_path", default="DPT/input", help="folder with input images"
    )

    parser.add_argument(
        "-o",
        "--output_path",
        default="DPT/output_monodepth",
        help="folder for output images",
    )

    parser.add_argument(
        "-m", "--model_weights", default=None, help="path to model weights"
    )

    parser.add_argument(
        "-t",
        "--model_type",
        default="dpt_hybrid",
        help="model type [dpt_large|dpt_hybrid|midas_v21]",
    )

    parser.add_argument("--kitti_crop", dest="kitti_crop", action="store_true")
    parser.add_argument("--absolute_depth", dest="absolute_depth", action="store_true")

    parser.add_argument("--optimize", dest="optimize", action="store_true")
    parser.add_argument("--no-optimize", dest="optimize", action="store_false")

    parser.set_defaults(optimize=True)
    parser.set_defaults(kitti_crop=False)
    parser.set_defaults(absolute_depth=False)

    args = parser.parse_args()

    default_models = {
        "midas_v21": "DPT/weights/midas_v21-f6b98070.pt",
        "dpt_large": "DPT/weights/dpt_large-midas-2f21e586.pt",
        "dpt_hybrid": "DPT/weights/dpt_hybrid-midas-501f0c75.pt",
        "dpt_hybrid_kitti": "DPT/weights/dpt_hybrid_kitti-cb926ef4.pt",
        "dpt_hybrid_nyu": "DPT/weights/dpt_hybrid_nyu-2ce69ec7.pt",
    }

    if args.model_weights is None:
        args.model_weights = default_models[args.model_type]

    # set torch options
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # compute depth maps
    run(
        args.input_path,
        args.output_path,
        args.model_weights,
        args.train_bool,
        args.model_type,
        args.optimize,
    )
