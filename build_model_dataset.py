import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw

SAM_VERSION = "sam3.1"
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Large-hf"
DINO_MODEL = "facebook/dinov3-vith16plus-pretrain-lvd1689m"
INPUT_CSV = Path("data/dataset.csv")
OUTPUT_CSV = Path("data/model_dataset.csv")
IMAGE_OUTPUT = Path("data/model_outputs")
SAM_PROMPTS = ["river", "snow", "vegetation", "sky", "ground"]
COLORS = [(0, 130, 255), (255, 80, 200), (0, 220, 80), (255, 210, 0), (255, 80, 50)]


def images(table):
    for path in table["image_file"]:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        with Image.open(path) as image:
            yield path, image.convert("RGB")


def release(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def add_sam3(table, prompts, device, folder):
    try:
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model, download_ckpt_from_hf
    except ImportError as error:
        raise SystemExit("SAM 3 is not installed") from error

    checkpoint = download_ckpt_from_hf(version=SAM_VERSION)
    model = build_sam3_image_model(
        device=device, checkpoint_path=checkpoint, load_from_HF=False
    )
    processor = Sam3Processor(model)
    results = {f"sam_{key}_count": [] for _, key in prompts}
    results |= {f"sam_{key}_coverage": [] for _, key in prompts}
    results |= {f"sam_{key}_score": [] for _, key in prompts}
    results["sam_image"] = []
    folder.mkdir(parents=True, exist_ok=True)

    for number, (source, image) in enumerate(images(table), 1):
        overlay = np.asarray(image.convert("L").convert("RGB")).copy()
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
            state = processor.set_image(image)
            for prompt_number, (prompt, key) in enumerate(prompts):
                output = processor.set_text_prompt(state=state, prompt=prompt)
                masks, scores = output["masks"], output["scores"]
                coverage = masks.bool().any(dim=0).float().mean().item() if len(masks) else 0
                mask = masks.bool().any(dim=0).squeeze().cpu().numpy() if len(masks) else np.zeros((image.height, image.width), bool)
                color = np.array(COLORS[prompt_number % len(COLORS)])
                overlay[mask] = (.55 * overlay[mask] + .45 * color).astype(np.uint8)
                results[f"sam_{key}_count"].append(len(masks))
                results[f"sam_{key}_coverage"].append(coverage)
                results[f"sam_{key}_score"].append(scores.float().mean().item() if len(scores) else 0)

        combined = Image.fromarray(overlay)
        draw = ImageDraw.Draw(combined)
        draw.rectangle((8, 8, 180, 16 + 20 * len(prompts)), fill="black")
        for i, (prompt, _) in enumerate(prompts):
            y = 14 + 20 * i
            draw.rectangle((14, y, 26, y + 12), fill=COLORS[i % len(COLORS)])
            draw.text((32, y - 2), prompt, fill="white")
        output_path = folder / f"{source.stem}_segments.jpg"
        combined.save(output_path, quality=90)
        results["sam_image"].append(str(output_path))
        print(f"SAM 3: {number}/{len(table)}")

    release(model)
    return table.assign(**results)


def add_depth(table, device, folder):
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL)
    model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL).to(device).eval()
    rows = []
    folder.mkdir(parents=True, exist_ok=True)
    for number, (source, image) in enumerate(images(table), 1):
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.inference_mode():
            output = model(**inputs)
            depth = processor.post_process_depth_estimation(
                output, target_sizes=[(image.height, image.width)]
            )[0]["predicted_depth"].float()
        flat = depth.flatten()
        q10, median, q90 = torch.quantile(flat, torch.tensor([.1, .5, .9], device=device))
        values = depth.cpu().numpy()
        gray = ((values - values.min()) / max(np.ptp(values), 1e-8) * 255).astype(np.uint8)
        output_path = folder / f"{source.stem}_depth.png"
        Image.fromarray(gray).save(output_path)
        rows.append((flat.mean().item(), flat.std().item(), q10.item(), median.item(), q90.item(), str(output_path)))
        print(f"Depth Anything: {number}/{len(table)}")

    release(model)
    columns = ["depth_mean", "depth_std", "depth_p10", "depth_median", "depth_p90", "depth_image"]
    return table.join(pd.DataFrame(rows, columns=columns, index=table.index))


def add_dino(table, device):
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(DINO_MODEL)
    model = AutoModel.from_pretrained(DINO_MODEL).to(device).eval()
    rows = []
    for number, (_, image) in enumerate(images(table), 1):
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.inference_mode():
            output = model(**inputs)
            embedding = getattr(output, "pooler_output", None)
            if embedding is None:
                embedding = output.last_hidden_state.mean(dim=1)
        rows.append(embedding[0].float().cpu().numpy())
        print(f"DINOv3: {number}/{len(table)}")

    release(model)
    features = pd.DataFrame(rows, index=table.index)
    features.columns = [f"dino_{i:04d}" for i in range(features.shape[1])]
    return table.join(features)


def main():
    table = pd.read_csv(INPUT_CSV)
    prompts = [(prompt, prompt) for prompt in SAM_PROMPTS]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Processing {len(table)} images on {device}")

    table = add_sam3(table, prompts, device, IMAGE_OUTPUT / "sam3")
    table = add_depth(table, device, IMAGE_OUTPUT / "depth")
    table = add_dino(table, device)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(table)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
