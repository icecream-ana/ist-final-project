# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Low-VRAM decoder fine-tuning.

This variant avoids instantiating the full LatentDiffusion model. It builds only
the first-stage AutoencoderKL from the LDM config and loads the
`first_stage_model.*` weights from a Stable Diffusion checkpoint.
"""

import argparse
import gc
import json
import os
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torchvision import transforms
from torchvision.utils import save_image

import utils
import utils_img
import utils_model

sys.path.append("src")
from ldm.models.autoencoder import AutoencoderKL
from loss.loss_provider import LossProvider


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def get_parser():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        group.add_argument(*args, **kwargs)

    group = parser.add_argument_group("Data parameters")
    aa("--train_dir", type=str, required=True)
    aa("--val_dir", type=str, required=True)
    aa("--val_num_imgs", type=int, default=None)
    aa("--num_workers", type=int, default=0)

    group = parser.add_argument_group("Model parameters")
    aa("--ldm_config", type=str, required=True)
    aa("--ldm_ckpt", type=str, required=True)
    aa("--msg_decoder_path", type=str, default="models/dec_48b_whit.torchscript.pt")
    aa("--num_bits", type=int, default=48)
    aa("--redundancy", type=int, default=1)
    aa("--decoder_depth", type=int, default=8)
    aa("--decoder_channels", type=int, default=64)

    group = parser.add_argument_group("Training parameters")
    aa("--batch_size", type=int, default=1)
    aa("--img_size", type=int, default=256)
    aa("--loss_i", type=str, default="watson-vgg")
    aa("--loss_w", type=str, default="bce")
    aa("--aux_device", type=str, default="cuda", choices=["cuda", "cpu"])
    aa("--lambda_i", type=float, default=0.2)
    aa("--lambda_w", type=float, default=1.0)
    aa("--optimizer", type=str, default="AdamW,lr=5e-4")
    aa("--steps", type=int, default=100)
    aa("--warmup_steps", type=int, default=20)

    group = parser.add_argument_group("Logging and saving freq. parameters")
    aa("--log_freq", type=int, default=10)
    aa("--save_img_freq", type=int, default=1000)

    group = parser.add_argument_group("Experiments parameters")
    aa("--num_keys", type=int, default=1)
    aa("--output_dir", type=str, default="output_lowvram/")
    aa("--seed", type=int, default=0)
    aa("--debug", type=utils.bool_inst, default=False)

    return parser


def load_first_stage_from_config(config_path, ckpt_path):
    print(f">>> Building first-stage autoencoder from {config_path}...")
    config = OmegaConf.load(config_path)
    first_stage_config = config.model.params.first_stage_config
    ae: AutoencoderKL = utils_model.instantiate_from_config(first_stage_config)

    print(f">>> Loading first_stage_model weights from {ckpt_path}...")
    pl_sd = torch.load(ckpt_path, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd.get("state_dict", pl_sd)
    ae_sd = {
        k.replace("first_stage_model.", "", 1): v
        for k, v in sd.items()
        if k.startswith("first_stage_model.")
    }
    if not ae_sd:
        raise RuntimeError("No first_stage_model.* weights found in checkpoint.")

    missing, unexpected = ae.load_state_dict(ae_sd, strict=False)
    print(f">>> Loaded autoencoder. missing={len(missing)} unexpected={len(unexpected)}")
    del pl_sd, sd, ae_sd
    gc.collect()

    ae.eval()
    ae.to(device)
    return ae


def build_msg_decoder(params, aux_device):
    print(f">>> Building hidden decoder with weights from {params.msg_decoder_path}...")
    if "torchscript" in params.msg_decoder_path:
        msg_decoder = torch.jit.load(params.msg_decoder_path).to(aux_device)
    else:
        msg_decoder = utils_model.get_hidden_decoder(
            num_bits=params.num_bits,
            redundancy=params.redundancy,
            num_blocks=params.decoder_depth,
            channels=params.decoder_channels,
        ).to(aux_device)
        ckpt = utils_model.get_hidden_decoder_ckpt(params.msg_decoder_path)
        print(msg_decoder.load_state_dict(ckpt, strict=False))
    msg_decoder.eval()
    return msg_decoder


def build_losses(params, aux_device):
    print(">>> Creating losses...")
    print(f"Losses: {params.loss_w} and {params.loss_i}...")
    if params.loss_w == "mse":
        loss_w = lambda decoded, keys, temp=10.0: torch.mean((decoded * temp - (2 * keys - 1)) ** 2)
    elif params.loss_w == "bce":
        loss_w = lambda decoded, keys, temp=10.0: F.binary_cross_entropy_with_logits(
            decoded * temp, keys, reduction="mean"
        )
    else:
        raise NotImplementedError

    if params.loss_i == "mse":
        loss_i = lambda imgs_w, imgs: torch.mean((imgs_w - imgs) ** 2)
    elif params.loss_i == "watson-dft":
        provider = LossProvider()
        loss_percep = provider.get_loss_function("Watson-DFT", colorspace="RGB", pretrained=True, reduction="sum")
        loss_percep = loss_percep.to(aux_device)
        loss_i = lambda imgs_w, imgs: loss_percep((1 + imgs_w) / 2.0, (1 + imgs) / 2.0) / imgs_w.shape[0]
    elif params.loss_i == "watson-vgg":
        provider = LossProvider()
        loss_percep = provider.get_loss_function("Watson-VGG", colorspace="RGB", pretrained=True, reduction="sum")
        loss_percep = loss_percep.to(aux_device)
        loss_i = lambda imgs_w, imgs: loss_percep((1 + imgs_w) / 2.0, (1 + imgs) / 2.0) / imgs_w.shape[0]
    elif params.loss_i == "ssim":
        provider = LossProvider()
        loss_percep = provider.get_loss_function("SSIM", colorspace="RGB", pretrained=True, reduction="sum")
        loss_percep = loss_percep.to(aux_device)
        loss_i = lambda imgs_w, imgs: loss_percep((1 + imgs_w) / 2.0, (1 + imgs) / 2.0) / imgs_w.shape[0]
    else:
        raise NotImplementedError
    return loss_w, loss_i


def train_lowvram(data_loader, optimizer, loss_w, loss_i, ldm_ae, ldm_decoder, msg_decoder, vqgan_to_imnet, key, params, aux_device):
    header = "Train"
    metric_logger = utils.MetricLogger(delimiter="  ")
    ldm_decoder.decoder.train()
    base_lr = optimizer.param_groups[0]["lr"]
    for ii, imgs in enumerate(metric_logger.log_every(data_loader, params.log_freq, header)):
        imgs = imgs.to(device)
        keys = key.repeat(imgs.shape[0], 1).to(aux_device)

        utils.adjust_learning_rate(optimizer, ii, params.steps, params.warmup_steps, base_lr)
        with torch.no_grad():
            imgs_z = ldm_ae.encode(imgs).mode()
            imgs_d0 = ldm_ae.decode(imgs_z)

        imgs_w = ldm_decoder.decode(imgs_z)
        decoded = msg_decoder(vqgan_to_imnet(imgs_w).to(aux_device))

        lossw = loss_w(decoded, keys)
        lossi = loss_i(imgs_w.to(aux_device), imgs_d0.to(aux_device))
        loss = params.lambda_w * lossw + params.lambda_i * lossi

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        with torch.no_grad():
            diff = ~torch.logical_xor(decoded > 0, keys > 0)
            bit_accs = torch.sum(diff, dim=-1) / diff.shape[-1]
            word_accs = bit_accs == 1
            log_stats = {
                "iteration": ii,
                "loss": loss.item(),
                "loss_w": lossw.item(),
                "loss_i": lossi.item(),
                "psnr": utils_img.psnr(imgs_w, imgs_d0).mean().item(),
                "bit_acc_avg": torch.mean(bit_accs).item(),
                "word_acc_avg": torch.mean(word_accs.type(torch.float)).item(),
                "lr": optimizer.param_groups[0]["lr"],
            }
        for name, value in log_stats.items():
            metric_logger.update(**{name: value})
        if ii % params.log_freq == 0:
            print(json.dumps(log_stats))

        if ii % params.save_img_freq == 0:
            save_image(torch.clamp(utils_img.unnormalize_vqgan(imgs), 0, 1), os.path.join(params.imgs_dir, f"{ii:03}_train_orig.png"), nrow=8)
            save_image(torch.clamp(utils_img.unnormalize_vqgan(imgs_d0), 0, 1), os.path.join(params.imgs_dir, f"{ii:03}_train_d0.png"), nrow=8)
            save_image(torch.clamp(utils_img.unnormalize_vqgan(imgs_w), 0, 1), os.path.join(params.imgs_dir, f"{ii:03}_train_w.png"), nrow=8)

        del imgs, keys, imgs_z, imgs_d0, imgs_w, decoded, lossw, lossi, loss
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("Averaged train stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def val_lowvram(data_loader, ldm_ae, ldm_decoder, msg_decoder, vqgan_to_imnet, key, params, aux_device):
    header = "Eval"
    metric_logger = utils.MetricLogger(delimiter="  ")
    ldm_decoder.decoder.eval()
    for ii, imgs in enumerate(metric_logger.log_every(data_loader, params.log_freq, header)):
        imgs = imgs.to(device)
        imgs_z = ldm_ae.encode(imgs).mode()
        imgs_d0 = ldm_ae.decode(imgs_z)
        imgs_w = ldm_decoder.decode(imgs_z)
        keys = key.repeat(imgs.shape[0], 1).to(aux_device)

        log_stats = {
            "iteration": ii,
            "psnr": utils_img.psnr(imgs_w, imgs_d0).mean().item(),
        }
        attacks = {
            "none": lambda x: x,
            "crop_01": lambda x: utils_img.center_crop(x, 0.1),
            "crop_05": lambda x: utils_img.center_crop(x, 0.5),
            "rot_25": lambda x: utils_img.rotate(x, 25),
            "rot_90": lambda x: utils_img.rotate(x, 90),
            "resize_03": lambda x: utils_img.resize(x, 0.3),
            "resize_07": lambda x: utils_img.resize(x, 0.7),
            "brightness_1p5": lambda x: utils_img.adjust_brightness(x, 1.5),
            "brightness_2": lambda x: utils_img.adjust_brightness(x, 2),
            "jpeg_80": lambda x: utils_img.jpeg_compress(x, 80),
            "jpeg_50": lambda x: utils_img.jpeg_compress(x, 50),
        }
        for name, attack in attacks.items():
            imgs_aug = attack(vqgan_to_imnet(imgs_w)).to(aux_device)
            decoded = msg_decoder(imgs_aug)
            diff = ~torch.logical_xor(decoded > 0, keys > 0)
            bit_accs = torch.sum(diff, dim=-1) / diff.shape[-1]
            word_accs = bit_accs == 1
            log_stats[f"bit_acc_{name}"] = torch.mean(bit_accs).item()
            log_stats[f"word_acc_{name}"] = torch.mean(word_accs.type(torch.float)).item()
        for name, value in log_stats.items():
            metric_logger.update(**{name: value})

        if ii % params.save_img_freq == 0:
            save_image(torch.clamp(utils_img.unnormalize_vqgan(imgs), 0, 1), os.path.join(params.imgs_dir, f"{ii:03}_val_orig.png"), nrow=8)
            save_image(torch.clamp(utils_img.unnormalize_vqgan(imgs_d0), 0, 1), os.path.join(params.imgs_dir, f"{ii:03}_val_d0.png"), nrow=8)
            save_image(torch.clamp(utils_img.unnormalize_vqgan(imgs_w), 0, 1), os.path.join(params.imgs_dir, f"{ii:03}_val_w.png"), nrow=8)

        del imgs, imgs_z, imgs_d0, imgs_w, keys
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("Averaged eval stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def main(params):
    torch.manual_seed(params.seed)
    torch.cuda.manual_seed_all(params.seed)
    np.random.seed(params.seed)

    print("__git__:{}".format(utils.get_sha()))
    print("__log__:{}".format(json.dumps(vars(params))))

    os.makedirs(params.output_dir, exist_ok=True)
    params.imgs_dir = os.path.join(params.output_dir, "imgs")
    os.makedirs(params.imgs_dir, exist_ok=True)

    aux_device = torch.device(params.aux_device if params.aux_device == "cpu" or torch.cuda.is_available() else "cpu")
    ldm_ae = load_first_stage_from_config(params.ldm_config, params.ldm_ckpt)
    msg_decoder = build_msg_decoder(params, aux_device)
    nbit = msg_decoder(torch.zeros(1, 3, 128, 128).to(aux_device)).shape[-1]

    for param in [*msg_decoder.parameters(), *ldm_ae.parameters()]:
        param.requires_grad = False

    print(f">>> Loading data from {params.train_dir} and {params.val_dir}...")
    vqgan_transform = transforms.Compose(
        [
            transforms.Resize(params.img_size),
            transforms.CenterCrop(params.img_size),
            transforms.ToTensor(),
            utils_img.normalize_vqgan,
        ]
    )
    train_loader = utils.get_dataloader(
        params.train_dir,
        vqgan_transform,
        params.batch_size,
        num_imgs=params.batch_size * params.steps,
        shuffle=True,
        num_workers=params.num_workers,
        collate_fn=None,
    )
    val_count = len(utils.get_image_paths(params.val_dir))
    val_num_imgs = params.val_num_imgs if params.val_num_imgs is not None else val_count
    val_num_imgs = min(val_num_imgs, val_count)
    val_loader = utils.get_dataloader(
        params.val_dir,
        vqgan_transform,
        params.batch_size,
        num_imgs=val_num_imgs,
        shuffle=False,
        num_workers=params.num_workers,
        collate_fn=None,
    )
    vqgan_to_imnet = transforms.Compose([utils_img.unnormalize_vqgan, utils_img.normalize_img])
    loss_w, loss_i = build_losses(params, aux_device)

    for ii_key in range(params.num_keys):
        print(f"\n>>> Creating key with {nbit} bits...")
        key = torch.randint(0, 2, (1, nbit), dtype=torch.float32, device=aux_device)
        key_str = "".join([str(int(ii)) for ii in key.tolist()[0]])
        print(f"Key: {key_str}")

        ldm_decoder = deepcopy(ldm_ae)
        ldm_decoder.encoder = nn.Identity()
        ldm_decoder.quant_conv = nn.Identity()
        ldm_decoder.to(device)
        for param in ldm_decoder.parameters():
            param.requires_grad = True

        optim_params = utils.parse_params(params.optimizer)
        optimizer = utils.build_optimizer(model_params=ldm_decoder.parameters(), **optim_params)

        print(">>> Training...")
        train_stats = train_lowvram(
            train_loader,
            optimizer,
            loss_w,
            loss_i,
            ldm_ae,
            ldm_decoder,
            msg_decoder,
            vqgan_to_imnet,
            key,
            params,
            aux_device,
        )
        val_stats = val_lowvram(
            val_loader,
            ldm_ae,
            ldm_decoder,
            msg_decoder,
            vqgan_to_imnet,
            key,
            params,
            aux_device,
        )
        log_stats = {
            "key": key_str,
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"val_{k}": v for k, v in val_stats.items()},
        }
        save_dict = {
            "ldm_decoder": ldm_decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "params": params,
        }
        torch.save(save_dict, os.path.join(params.output_dir, f"checkpoint_{ii_key:03d}.pth"))
        with (Path(params.output_dir) / "log.txt").open("a") as f:
            f.write(json.dumps(log_stats) + "\n")
        with (Path(params.output_dir) / "keys.txt").open("a") as f:
            f.write(os.path.join(params.output_dir, f"checkpoint_{ii_key:03d}.pth") + "\t" + key_str + "\n")


if __name__ == "__main__":
    parser = get_parser()
    main(parser.parse_args())
