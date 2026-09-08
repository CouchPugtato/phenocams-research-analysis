import gc
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw

DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Large-hf"
DINO_MODEL = "facebook/dinov3-vith16plus-pretrain-lvd1689m"

INPUT_CSV = Path("data/dataset.csv")
OUTPUT_CSV = Path("data/model_dataset.csv")
IMAGE_OUTPUT = Path("data/model_outputs")

SAM_PROMPTS = ["river", "snow", "vegetation", "ground"]
SAM_CONFIDENCE = 0.2
COLORS = [(0, 130, 255), (255, 80, 200), (0, 220, 80), (255, 210, 0), (255, 80, 50)]

warnings.filterwarnings(
    "ignore",
    message="Importing from timm.models.layers is deprecated.*",
    category=FutureWarning,
)
CONSENSUS_MASK_THRESHOLD = 5


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
        from sam3.model_builder import build_sam3_image_model
    except ImportError as error:
        raise SystemExit("SAM 3 is not installed") from error

    model = build_sam3_image_model(device=device)
    processor = Sam3Processor(model, confidence_threshold=SAM_CONFIDENCE)
    results = {f"sam_{key}_coverage": [] for _, key in prompts}
    results |= {f"sam_{key}_score": [] for _, key in prompts}
    results["sam_image"] = []
    results["sam_river_mask"] = []
    river_masks = []
    folder.mkdir(parents=True, exist_ok=True)

    for number, (source, image) in enumerate(images(table), 1):
        overlay = np.asarray(image.convert("L").convert("RGB")).copy()
        river_mask = np.zeros((image.height, image.width), bool)
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
            state = processor.set_image(image)
            for prompt_number, (prompt, key) in enumerate(prompts):
                output = processor.set_text_prompt(state=state, prompt=prompt)
                masks, scores = output["masks"], output["scores"]
                coverage = masks.bool().any(dim=0).float().mean().item() if len(masks) else 0
                mask = masks.bool().any(dim=0).squeeze().cpu().numpy() if len(masks) else np.zeros((image.height, image.width), bool)
                color = np.array(COLORS[prompt_number % len(COLORS)])
                overlay[mask] = (.55 * overlay[mask] + .45 * color).astype(np.uint8)
                if key == "river":
                    river_mask = mask
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
        mask_path = folder / f"{source.stem}_river_mask.png"
        Image.fromarray((river_mask * 255).astype(np.uint8)).save(mask_path)
        river_masks.append(river_mask)
        results["sam_image"].append(str(output_path))
        results["sam_river_mask"].append(str(mask_path))
        print(f"SAM 3: {number}/{len(table)}")

    release(model)
    if len({mask.shape for mask in river_masks}) != 1:
        raise ValueError("All images must have the same dimensions to combine their river masks")
    combined_river = np.sum(river_masks, axis=0) >= CONSENSUS_MASK_THRESHOLD
    combined_path = folder / "all_images_river_mask.png"
    Image.fromarray((combined_river * 255).astype(np.uint8)).save(combined_path)
    results["sam_combined_river_mask"] = [str(combined_path)] * len(table)
    return table.assign(**results)


def depth_stats(values, prefix):
    if not values.size:
        return {f"{prefix}_{name}": np.nan for name in ("p10", "median", "p90")}
    q10, median, q90 = np.quantile(values, [.1, .5, .9])
    return {
        f"{prefix}_p10": q10, f"{prefix}_median": median, f"{prefix}_p90": q90,
    }


def align_depth(values, reference, excluded):
    stable = ~excluded[::8, ::8]
    current, target = values[::8, ::8][stable], reference[::8, ::8][stable]
    if current.size < 100:
        return values
    current_q = np.quantile(current, [.25, .5, .75])
    target_q = np.quantile(target, [.25, .5, .75])
    scale = (target_q[2] - target_q[0]) / max(current_q[2] - current_q[0], 1e-8)
    shift = target_q[1] - scale * current_q[1]
    return scale * values + shift


def add_depth(table, device, folder):
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL)
    model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL).to(device).eval()
    depth_maps, masks, image_paths, river_image_paths = [], [], [], []
    folder.mkdir(parents=True, exist_ok=True)
    mask_file = table["sam_combined_river_mask"].iloc[0]
    for number, (source, image) in enumerate(images(table), 1):
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.inference_mode():
            output = model(**inputs)
            depth = processor.post_process_depth_estimation(
                output, target_sizes=[(image.height, image.width)]
            )[0]["predicted_depth"].float()
        values = depth.cpu().numpy()
        with Image.open(mask_file) as mask_image:
            mask = np.asarray(mask_image.convert("L").resize(image.size, Image.Resampling.NEAREST)) > 0
        gray = ((values - values.min()) / max(np.ptp(values), 1e-8) * 255).astype(np.uint8)
        output_path = folder / f"{source.stem}_depth.png"
        river_output_path = folder / f"{source.stem}_depth_river.png"
        Image.fromarray(gray).save(output_path)
        Image.fromarray(np.where(mask, gray, 0).astype(np.uint8)).save(river_output_path)
        depth_maps.append(values)
        masks.append(mask)
        image_paths.append(str(output_path))
        river_image_paths.append(str(river_output_path))
        print(f"Depth Anything: {number}/{len(table)}")

    release(model)
    reference = depth_maps[0]

    rows = []
    for values, mask, image_path, river_image_path in zip(depth_maps, masks, image_paths, river_image_paths):
        aligned = align_depth(values, reference, mask)
        row = depth_stats(values.ravel(), "depth")
        row |= depth_stats(aligned[mask], "depth_river")
        row |= {
            "depth_image": image_path,
            "depth_river_image": river_image_path,
        }
        rows.append(row)
    medians = [row["depth_river_median"] for row in rows if np.isfinite(row["depth_river_median"])]
    average_median = np.mean(medians) if medians else np.nan
    for row in rows:
        row["depth_change_from_average"] = row["depth_river_median"] - average_median
    return table.join(pd.DataFrame(rows, index=table.index))


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
    table = pd.read_csv(INPUT_CSV).sort_values("image_timestamp").reset_index(drop=True)
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
