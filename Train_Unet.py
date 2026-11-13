import torch.optim
from tensorboardX import SummaryWriter
import os
import numpy as np
import random
import argparse
import time
import sys
from torch.backends import cudnn
from Load_Dataset import (
    ImageToImage2D,
    RandomGenerator,
    UltrasoundAxialGenerator,
    ValGenerator,
)
from UNet import UNet
from UNetPlusPlus import NestedUNet
from DeepLabv3plus import DeepLabV3Plus
from ResU_Net import ResUnet
from TransUNet import TransUNet
from SwinUNet import SwinUNet
from torch.utils.data import DataLoader
import logging
from Train_one_epoch import train_one_epoch
import UC_Config as config
from torchvision import transforms
from utils import CosineAnnealingWarmRestarts, LogitsDiceBCE, WeightedDiceBCE
#from E_SegNet_2D import E_SegNet_2D
#from ResU_Net import ResUnet
from SE_UNet import SE_UNet
from SE_UNet_MK_KAN_RPA import SE_UNet_MK_KAN_RPA
from SE_UNet_Ablation import (
    SE_UNet_MK,
    SE_UNet_KAN,
    SE_UNet_RPA,
    SE_UNet_MK_KAN,
    SE_UNet_MK_RPA,
    SE_UNet_KAN_RPA,
)
from  E_SE_SegNet import E_SE_SegNet_2D
from model_factory import create_model, FACTORY_MODEL_CHOICES


MODEL_CHOICES = [
    "UNet",
    "UNetPlusPlus",
    "UNet++",
    "DeepLabV3Plus",
    "DeepLabv3+",
    "ResUnet",
    "ResUNet",
    "SE_UNet",
    "SEUNet",
    "SwinUNet",
    "Swin-Unet",
    "TransUNet",
    "TransUnet",
    "SE_UNet_MK",
    "SE_UNet_KAN",
    "SE_UNet_RPA",
    "SE_UNet_MK_KAN",
    "SE_UNet_MK_RPA",
    "SE_UNet_KAN_RPA",
    "SE_UNet_MK_KAN_RPA",
    "SEUNet_MK_KAN_RPA",
    "SE_MK_KAN_RPA",
    "UCTransNet",
    "E_SE_SegNet_2D",
]
MODEL_CHOICES = sorted(set(MODEL_CHOICES + FACTORY_MODEL_CHOICES))

def logger_config(log_path):
    loggerr = logging.getLogger()
    loggerr.handlers.clear()
    loggerr.setLevel(level=logging.INFO)
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    loggerr.addHandler(handler)
    loggerr.addHandler(console)
    return loggerr


def save_checkpoint(state, save_path):
    '''
        Save the current model.
        If the model is the best model since beginning of the training
        it will be copy
    '''
    logger.info('\t Saving to {}'.format(save_path))
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    epoch = state['epoch']  # epoch no
    best_model = state['best_model']  # bool
    model = state['model']  # model type

    if state.get('final_model', False):
        filename = save_path + '/' + \
                   'final_model-{}.pth.tar'.format(model)
    elif best_model:
        filename = save_path + '/' + \
                   'best_model-{}.pth.tar'.format(model)
    else:
        filename = save_path + '/' + \
                   'model-{}-{:02d}.pth.tar'.format(model, epoch)
    torch.save(state, filename)


def worker_init_fn(worker_id):
    random.seed(config.seed + worker_id)


def load_transfer_checkpoint(model, checkpoint_path, trainable_pattern=None):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    source_state = checkpoint.get("state_dict", checkpoint)
    source_state = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in source_state.items()
    }
    target_state = model.state_dict()
    matched = {
        key: value for key, value in source_state.items()
        if key in target_state and target_state[key].shape == value.shape
    }
    shape_mismatch = [
        key for key, value in source_state.items()
        if key in target_state and target_state[key].shape != value.shape
    ]
    missing_old = [
        key for key in target_state
        if key not in matched and not (trainable_pattern and trainable_pattern in key)
    ]
    unexpected = [key for key in source_state if key not in target_state]
    if shape_mismatch or missing_old or unexpected:
        raise RuntimeError(
            "Transfer gate failed: shape_mismatch={}, missing_old={}, unexpected={}".format(
                shape_mismatch, missing_old, unexpected
            )
        )
    load_result = model.load_state_dict(matched, strict=False)
    expected_missing = [
        key for key in load_result.missing_keys
        if not (trainable_pattern and trainable_pattern in key)
    ]
    if expected_missing or load_result.unexpected_keys:
        raise RuntimeError(
            "Transfer load failed: missing={}, unexpected={}".format(
                expected_missing, load_result.unexpected_keys
            )
        )
    logger.info(
        "Transferred {}/{} original tensors from {}".format(
            len(matched), len(source_state), checkpoint_path
        )
    )

    if trainable_pattern:
        for name, parameter in model.named_parameters():
            parameter.requires_grad = trainable_pattern in name
        trainable_names = [
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        trainable_count = sum(
            parameter.numel() for parameter in model.parameters()
            if parameter.requires_grad
        )
        logger.info(
            "Frozen backbone; trainable pattern='{}', tensors={}, parameters={}".format(
                trainable_pattern, len(trainable_names), trainable_count
            )
        )


def keep_frozen_batchnorm_eval(model):
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            if not any(parameter.requires_grad for parameter in module.parameters()):
                module.eval()


##################################################################################
#=================================================================================
#          Main Loop: load model,
#=================================================================================
##################################################################################
def main_loop(batch_size=config.batch_size, model_type='', tensorboard=True,
              val_interval=0, train_min_delta=1e-4, min_epochs=12,
              data_parallel=False, init_checkpoint=None,
              trainable_pattern=None):
    # Load train and val data
    augmentation_mode = os.environ.get(
        "UC_TRAIN_AUGMENTATION",
        "legacy",
    ).strip().lower()
    if augmentation_mode == "ultrasound_axial":
        train_generator = UltrasoundAxialGenerator(
            output_size=[config.img_size, config.img_size],
        )
    elif augmentation_mode == "legacy":
        train_generator = RandomGenerator(
            output_size=[config.img_size, config.img_size],
        )
    else:
        raise ValueError(
            "UC_TRAIN_AUGMENTATION must be 'legacy' or "
            "'ultrasound_axial'"
        )
    logger.info("Training augmentation: {}".format(augmentation_mode))
    train_tf = transforms.Compose([train_generator])
    train_dataset = ImageToImage2D(config.train_dataset, train_tf, image_size=config.img_size)
    num_workers = int(os.environ.get("UC_NUM_WORKERS", "8"))
    pin_memory = os.environ.get("UC_PIN_MEMORY", "1") == "1"
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              worker_init_fn=worker_init_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory,
                              persistent_workers=num_workers > 0,
                              prefetch_factor=4 if num_workers > 0 else None)
    if val_interval > 0:
        val_tf = ValGenerator(output_size=[config.img_size, config.img_size])
        val_dataset = ImageToImage2D(config.val_dataset, val_tf, image_size=config.img_size)
        val_loader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                worker_init_fn=worker_init_fn,
                                num_workers=num_workers,
                                pin_memory=pin_memory,
                                persistent_workers=num_workers > 0,
                                prefetch_factor=4 if num_workers > 0 else None)
    else:
        val_loader = None

    lr = config.learning_rate
    logger.info(model_type)

    if model_type in FACTORY_MODEL_CHOICES:
        model = create_model(model_type, in_ch=config.n_channels, out_ch=config.n_labels)
    elif model_type == 'UNet':
        model = UNet(in_ch=config.n_channels, out_ch=config.n_labels)
    elif model_type in ('UNetPlusPlus', 'UNet++'):
        model = NestedUNet(in_channels=config.n_channels, num_classes=config.n_labels,
                           deep_supervision=True, bilinear=True)
    elif model_type in ('DeepLabV3Plus', 'DeepLabv3+'):
        model = DeepLabV3Plus(
            n_classes=config.n_labels,
            n_blocks=[3, 4, 6, 3],
            atrous_rates=[6, 12, 18],
            multi_grids=[1, 2, 4],
            output_stride=8,
        )
    elif model_type in ('ResUnet', 'ResUNet'):
        model = ResUnet(in_channels=config.n_channels, num_classes=config.n_labels,
                        base_c=32, bilinear=True)
    elif model_type in ('SE_UNet', 'SEUNet'):
        model = SE_UNet(in_ch=config.n_channels, out_ch=config.n_labels)
    elif model_type in ('SwinUNet', 'Swin-Unet'):
        model = SwinUNet(in_ch=config.n_channels, out_ch=config.n_labels, base_ch=32)
    elif model_type in ('TransUNet', 'TransUnet'):
        model = TransUNet(in_ch=config.n_channels, out_ch=config.n_labels, base_ch=32)
    elif model_type == 'SE_UNet_MK':
        model = SE_UNet_MK(in_ch=config.n_channels, out_ch=config.n_labels)
    elif model_type == 'SE_UNet_KAN':
        model = SE_UNet_KAN(in_ch=config.n_channels, out_ch=config.n_labels)
    elif model_type == 'SE_UNet_RPA':
        model = SE_UNet_RPA(in_ch=config.n_channels, out_ch=config.n_labels)
    elif model_type == 'SE_UNet_MK_KAN':
        model = SE_UNet_MK_KAN(in_ch=config.n_channels, out_ch=config.n_labels)
    elif model_type == 'SE_UNet_MK_RPA':
        model = SE_UNet_MK_RPA(in_ch=config.n_channels, out_ch=config.n_labels)
    elif model_type == 'SE_UNet_KAN_RPA':
        model = SE_UNet_KAN_RPA(in_ch=config.n_channels, out_ch=config.n_labels)
    elif model_type in ('SE_UNet_MK_KAN_RPA', 'SEUNet_MK_KAN_RPA', 'SE_MK_KAN_RPA'):
        model = SE_UNet_MK_KAN_RPA(in_ch=config.n_channels, out_ch=config.n_labels)
    elif model_type in ('UCTransNet', 'E_SE_SegNet_2D'):
        config_vit = config.get_CTranS_config()
        logger.info('transformer head num: {}'.format(config_vit.transformer.num_heads))
        logger.info('transformer layers num: {}'.format(config_vit.transformer.num_layers))
        logger.info('transformer expand ratio: {}'.format(config_vit.expand_ratio))
        #model = E_SegNet_2D(model_name="mobilenetv4_conv_medium.e500_r224_in1k", image_size=224)
        #model = ResUnet(in_channels=3, num_classes=1, base_c=32, bilinear=True)
        #model = SE_UNet(in_ch=3, out_ch=1)
        model = E_SE_SegNet_2D(model_name="mobilenetv4_conv_medium.e500_r224_in1k", image_size=224)




    elif model_type == 'UCTransNet_pretrain':
        config_vit = config.get_CTranS_config()
        logger.info('transformer head num: {}'.format(config_vit.transformer.num_heads))
        logger.info('transformer layers num: {}'.format(config_vit.transformer.num_layers))
        logger.info('transformer expand ratio: {}'.format(config_vit.expand_ratio))
        #model = E_SegNet_2D(model_name="mobilenetv4_conv_medium.e500_r224_in1k", image_size=224)
        #model = ResUnet(in_channels=3, num_classes=1, base_c=32, bilinear=True)
        #model = SE_UNet(in_ch=3, out_ch=1)
        model = E_SE_SegNet_2D(model_name="mobilenetv4_conv_medium.e500_r224_in1k", image_size=224)

        pretrained_UNet_model_path = "./nets/best_model-UNet.pth.tar"
        pretrained_UNet = torch.load(pretrained_UNet_model_path, map_location='cuda')
        pretrained_UNet = pretrained_UNet['state_dict']
        model2_dict = model.state_dict()
        state_dict = {k: v for k, v in pretrained_UNet.items() if k in model2_dict.keys()}
        print(state_dict.keys())
        model2_dict.update(state_dict)
        model.load_state_dict(model2_dict)
        logger.info('Load successful!')

    else:
        raise TypeError('Please enter a valid name for the model type')

    if init_checkpoint:
        load_transfer_checkpoint(model, init_checkpoint, trainable_pattern)

    model = model.cuda()
    if data_parallel and torch.cuda.device_count() > 1:
        logger.info("Using DataParallel on {} GPUs".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
    if os.environ.get("UC_CHANNELS_LAST", "1") == "1":
        model = model.to(memory_format=torch.channels_last)
    # if torch.cuda.device_count() > 1:
    #     print ("Let's use {0} GPUs!".format(torch.cuda.device_count()))
    # model = nn.DataParallel(model, device_ids=[0])
    if os.environ.get("UC_LOSS_MODE", "weighted_probability") == "logits_dice_bce":
        criterion = LogitsDiceBCE(dice_weight=0.5, BCE_weight=0.5)
        logger.info("Using stable raw-logits Dice+BCE loss")
    else:
        criterion = WeightedDiceBCE(dice_weight=0.5, BCE_weight=0.5)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)  # Choose optimize
    if config.cosineLR is True and os.environ.get("UC_DISABLE_LR_SCHEDULER", "0") != "1":
        lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-4)
    else:
        lr_scheduler = None
    if tensorboard:
        log_dir = config.tensorboard_folder
        logger.info('log dir: {}'.format(log_dir))
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        writer = SummaryWriter(log_dir)
    else:
        writer = None

    max_dice = 0.0
    best_epoch = 1
    best_train_loss = float('inf')
    train_stopping_count = 0
    final_epoch = 0
    loss_min = 99
    iou_max = 0
    dice_max = 0
    loss_min1 = 99
    iou_max1 = 0
    dice_max1 = 0
    for epoch in range(config.epochs):  # loop over the dataset multiple times
        logger.info('\n========= Epoch [{}/{}] ========='.format(epoch + 1, config.epochs))
        logger.info(config.session_name)
        # train for one epoch
        model.train(True)
        if trainable_pattern:
            keep_frozen_batchnorm_eval(model)
        logger.info('Training with batch size : {}'.format(batch_size))
        train_scheduler = lr_scheduler if val_interval == 0 else None
        A, B, C, D, E = train_one_epoch(train_loader, model, criterion, optimizer, writer, epoch, train_scheduler, model_type,
                                        logger, loss_min, iou_max, dice_max)
        loss_min, iou_max, dice_max = C, D, E
        final_epoch = epoch
        train_loss = A.item() if hasattr(A, "item") else float(A)
        if best_train_loss - train_loss > train_min_delta:
            best_train_loss = train_loss
            train_stopping_count = 0
        else:
            train_stopping_count += 1
        logger.info('\t train_loss_best: {:.4f}, train_stopping_count: {}/{}'.format(
            best_train_loss, train_stopping_count, config.early_stopping_patience))

        should_validate = val_loader is not None and (
            (epoch + 1) % val_interval == 0 or epoch + 1 == config.epochs
        )
        if should_validate:
            logger.info('Validation')
            with torch.no_grad():
                model.eval()
                val_loss, val_dice, C1, D1, E1 = train_one_epoch(val_loader, model, criterion,
                                                                 optimizer, writer, epoch, lr_scheduler, model_type, logger,
                                                                 loss_min1, iou_max1, dice_max1)
            loss_min1, iou_max1, dice_max1 = C1, D1, E1
            if val_dice > max_dice:
                logger.info(
                    '\t Saving best model, mean dice increased from: {:.4f} to {:.4f}'.format(max_dice, val_dice))
                max_dice = val_dice
                best_epoch = epoch + 1
                save_checkpoint({'epoch': epoch,
                                 'best_model': True,
                                 'model': model_type,
                                 'state_dict': model.state_dict(),
                                 'val_loss': val_loss,
                                 'optimizer': optimizer.state_dict()}, config.model_path)
            else:
                logger.info('\t Mean dice:{:.4f} does not increase, '
                            'the best is still: {:.4f} in epoch {}'.format(val_dice, max_dice, best_epoch))

        if epoch + 1 >= min_epochs and train_stopping_count > config.early_stopping_patience:
            logger.info('\t train loss converged, early stopping!')
            break

    save_checkpoint({'epoch': final_epoch,
                     'best_model': False,
                     'final_model': True,
                     'model': model_type,
                     'state_dict': model.state_dict(),
                     'train_loss': loss_min,
                     'optimizer': optimizer.state_dict()}, config.model_path)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=config.model_name, choices=MODEL_CHOICES)
    parser.add_argument("--batch-size", type=int, default=config.batch_size)
    parser.add_argument("--epochs", type=int, default=config.epochs)
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--early-stopping", type=int, default=config.early_stopping_patience)
    parser.add_argument("--min-epochs", type=int, default=12)
    parser.add_argument("--train-min-delta", type=float, default=1e-4)
    parser.add_argument("--val-interval", type=int, default=0)
    parser.add_argument("--vis-frequency", type=int, default=config.vis_frequency)
    parser.add_argument("--num-workers", type=int, default=int(os.environ.get("UC_NUM_WORKERS", "4")))
    parser.add_argument("--tensorboard", action="store_true", default=False)
    parser.add_argument("--data-parallel", action="store_true", default=False)
    parser.add_argument("--session-name", default=None)
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--trainable-pattern", default=None)
    parser.add_argument("--train-dataset", default=None)
    parser.add_argument("--val-dataset", default=None)
    args = parser.parse_args()

    skip_file = os.path.join(os.path.dirname(__file__), "experiment_scripts", "skip_models.txt")
    if os.path.exists(skip_file):
        with open(skip_file, "r", encoding="utf-8") as f:
            skip_models = {line.strip() for line in f if line.strip() and not line.strip().startswith("#")}
        if args.model in skip_models:
            print("Skipping model by skip list: {}".format(args.model))
            sys.exit(3)

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    config.model_name = args.model
    config.save_name = args.model
    config.batch_size = args.batch_size
    config.epochs = args.epochs
    config.early_stopping_patience = args.early_stopping
    config.vis_frequency = args.vis_frequency
    if args.train_dataset:
        config.train_dataset = args.train_dataset
    if args.val_dataset:
        config.val_dataset = args.val_dataset
    os.environ["UC_NUM_WORKERS"] = str(args.num_workers)
    config.session_name = args.session_name or ('Fast_session' + '_' + time.strftime('%m.%d_%Hh%M'))
    config.save_path = config.task_name + '/' + config.save_name + '/' + config.session_name + '/'
    config.model_path = config.save_path + 'models/'
    config.tensorboard_folder = config.save_path + 'tensorboard_logs/'
    config.logger_path = config.save_path + config.session_name + ".log"
    config.visualize_path = config.save_path + 'visualize_val/'

    deterministic = os.environ.get("UC_DETERMINISTIC", "0") == "1"
    if not deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    if not os.path.isdir(config.save_path):
        os.makedirs(config.save_path)

    logger = logger_config(log_path=config.logger_path)
    #print(config.model_name)
    #print(config.batch_size)
    model = main_loop(batch_size=config.batch_size, model_type=config.model_name, tensorboard=args.tensorboard,
                      val_interval=args.val_interval, train_min_delta=args.train_min_delta,
                      min_epochs=args.min_epochs, data_parallel=args.data_parallel,
                      init_checkpoint=args.init_checkpoint,
                      trainable_pattern=args.trainable_pattern)
