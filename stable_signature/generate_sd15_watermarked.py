import argparse
import os
import sys
from pathlib import Path
from types import MethodType

import torch
from omegaconf import OmegaConf

import utils_model

sys.path.append("src")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ldm_config", type=str, required=True)
    parser.add_argument("--ldm_ckpt", type=str, required=True)
    parser.add_argument("--wm_decoder_ckpt", type=str, required=True)
    parser.add_argument("--model", type=str, default=None, help="Diffusers model dir or HF model id.")
    parser.add_argument("--single_file", type=str, default=None, help="Local .ckpt/.safetensors for diffusers from_single_file.")
    parser.add_argument("--out_dir_w", type=str, default="outputs/generated_w")
    parser.add_argument("--out_dir_nw", type=str, default="outputs/generated_nw")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--prompts_file", type=str, default=None)
    parser.add_argument("--num_images", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, choices=["float32", "float16"], default="float16")
    parser.add_argument("--attention_slicing", action="store_true")
    parser.add_argument("--vae_slicing", action="store_true")
    return parser


def read_prompts(args):
    if args.prompts_file:
        prompts = [
            line.strip()
            for line in Path(args.prompts_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    elif args.prompt:
        prompts = [args.prompt]
    else:
        prompts = [
            "a photo of a cat drinking water",
            "a landscape photo of mountains at sunrise",
            "a red car parked on a city street",
            "a bowl of fresh fruit on a wooden table",
            "a small house near a lake",
            "a dog running through a field",
            "a coffee cup beside a notebook",
            "a street market on a rainy evening",
        ]

    if not prompts:
        raise ValueError("No prompts provided.")
    return [prompts[i % len(prompts)] for i in range(args.num_images)]


def load_first_stage(config_path, ckpt_path, device, dtype):
    config = OmegaConf.load(config_path)
    ae = utils_model.instantiate_from_config(config.model.params.first_stage_config)
    pl_sd = torch.load(ckpt_path, map_location="cpu")
    sd = pl_sd.get("state_dict", pl_sd)
    ae_sd = {
        k.replace("first_stage_model.", "", 1): v
        for k, v in sd.items()
        if k.startswith("first_stage_model.")
    }
    if not ae_sd:
        raise RuntimeError("No first_stage_model.* weights found in ldm checkpoint.")
    missing, unexpected = ae.load_state_dict(ae_sd, strict=False)
    print(f"Loaded original first-stage model: missing={len(missing)} unexpected={len(unexpected)}")
    ae.eval().to(device=device, dtype=dtype)
    return ae


def load_watermarked_decoder(ae, wm_decoder_ckpt):
    ckpt = torch.load(wm_decoder_ckpt, map_location="cpu")
    state_dict = ckpt["ldm_decoder"] if "ldm_decoder" in ckpt else ckpt
    missing, unexpected = ae.load_state_dict(state_dict, strict=False)
    print(f"Loaded watermarked decoder: missing={len(missing)} unexpected={len(unexpected)}")
    decoder_unexpected = [k for k in unexpected if k.startswith("decoder.")]
    if decoder_unexpected:
        raise RuntimeError(f"Decoder keys were not loaded correctly: {decoder_unexpected[:10]}")
    ae.eval()
    return ae


def build_pipeline(args, dtype):
    from diffusers import StableDiffusionPipeline

    common_kwargs = {
        "torch_dtype": dtype,
        "safety_checker": None,
        "requires_safety_checker": False,
    }
    original_cuda_is_available = torch.cuda.is_available
    if args.device == "cpu":
        # diffusers' ckpt converter chooses CUDA internally when it is visible.
        # Force CPU conversion/loading for machines where CUDA exists but is too small.
        torch.cuda.is_available = lambda: False
    try:
        if args.single_file:
            try:
                pipe = StableDiffusionPipeline.from_single_file(
                    args.single_file,
                    original_config_file=args.ldm_config,
                    device="cpu",
                    load_safety_checker=False,
                    **common_kwargs,
                )
            except TypeError:
                pipe = StableDiffusionPipeline.from_single_file(
                    args.single_file,
                    device="cpu",
                    load_safety_checker=False,
                    **common_kwargs,
                )
        elif args.model:
            pipe = StableDiffusionPipeline.from_pretrained(args.model, **common_kwargs)
        else:
            raise ValueError("Provide either --model or --single_file.")
    finally:
        torch.cuda.is_available = original_cuda_is_available

    pipe = pipe.to(args.device)
    if args.attention_slicing:
        pipe.enable_attention_slicing()
    if args.vae_slicing:
        pipe.enable_vae_slicing()
    pipe.set_progress_bar_config(disable=False)
    return pipe


def patch_pipe_decode(pipe, ae):
    try:
        from diffusers.models.autoencoders.vae import DecoderOutput
    except Exception:
        DecoderOutput = None

    def decode(self, z, return_dict=True, *args, **kwargs):
        sample = ae.decode(z)
        if not return_dict:
            return (sample,)
        if DecoderOutput is not None:
            return DecoderOutput(sample=sample)
        return {"sample": sample}

    pipe.vae.decode = MethodType(decode, pipe.vae)


def generate_set(pipe, prompts, out_dir, args):
    os.makedirs(out_dir, exist_ok=True)
    for i, prompt in enumerate(prompts):
        generator = torch.Generator(device=args.device).manual_seed(args.seed + i)
        image = pipe(
            prompt,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            height=args.height,
            width=args.width,
            generator=generator,
        ).images[0]
        image.save(os.path.join(out_dir, f"{i:05d}.png"))
        print(f"Saved {out_dir}/{i:05d}.png")


def main():
    args = get_parser().parse_args()
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    prompts = read_prompts(args)

    print(">>> Loading diffusion pipeline...")
    pipe = build_pipeline(args, dtype)

    print(">>> Generating non-watermarked images...")
    generate_set(pipe, prompts, args.out_dir_nw, args)

    print(">>> Loading watermarked LDM decoder...")
    if hasattr(pipe, "vae"):
        pipe.vae.to("cpu")
        if args.device == "cuda":
            torch.cuda.empty_cache()
    ae = load_first_stage(args.ldm_config, args.ldm_ckpt, args.device, dtype)
    ae = load_watermarked_decoder(ae, args.wm_decoder_ckpt)
    patch_pipe_decode(pipe, ae)

    print(">>> Generating watermarked images...")
    generate_set(pipe, prompts, args.out_dir_w, args)

    print("Done.")


if __name__ == "__main__":
    main()
