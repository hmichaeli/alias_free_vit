#!/usr/bin/env python3
"""
Demo utilities for XCiT model robustness comparison.

This module provides functionality to evaluate model robustness to:
1. Crop shifts - discrete pixel shifts
2. Bilinear fractional shifts - sub-pixel shifts using bilinear interpolation

The demo uses a subset of ImageNet images from the imagenet-sample-images repository.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from tqdm import tqdm

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))
from timm.models import create_model

# register pretrained models
import pretrained_models  # noqa: F401

# Load ImageNet class names


def load_imagenet_classes():
    """Load ImageNet class names from the JSON file in the repository."""
    # First try to load from the cloned repository
    json_file = (
        Path(__file__).parent
        / "Small-ImageNet-Validation-Dataset-1000-Classes"
        / "imagenet_class_index.json"
    )

    if json_file.exists():
        with open(json_file, "r", encoding="utf-8") as f:
            class_index = json.load(f)

        # Convert to list of class names, ordered by index
        classes = [""] * 1000  # Initialize with empty strings
        for idx_str, (_, class_name) in class_index.items():
            idx = int(idx_str)
            classes[idx] = class_name

        return classes

    # Final fallback: create generic class names (avoid using experimental directory)
    return [f"class_{i}" for i in range(1000)]


# Global variable to store class names
IMAGENET_CLASSES = load_imagenet_classes()


def prettify_imagenet_label(label: str) -> str:
    """Return a nicer looking ImageNet label for display in GIFs.

    - Replace underscores with spaces
    - Capitalize only the first letter (keep the rest as-is)
    """
    if not isinstance(label, str):
        label = str(label)
    pretty = label.replace("_", " ")
    return pretty[:1].upper() + pretty[1:] if pretty else pretty


def clone_sample_images(
    dest_dir: Optional[Path] = None,
    repo_url: str = "https://github.com/hmichaeli/Small-ImageNet-Validation-Dataset-1000-Classes",
    update_if_exists: bool = True,
) -> Path:
    """Clone the Small-ImageNet-Validation-Dataset-1000-Classes repo used by the demo.

    Args:
        dest_dir: Destination directory to clone into. Defaults to
            demo/Small-ImageNet-Validation-Dataset-1000-Classes next to this file.
        repo_url: Git URL of the dataset repository.
        update_if_exists: If True and the destination already exists and is a git repo,
            fetch and fast-forward pull to update.

    Returns:
        Path to the local clone directory.
    """
    # Determine repository clone directory
    if dest_dir is None:
        base_dir = Path(__file__).parent
    else:
        base_dir = Path(dest_dir)
    # Always clone under a fixed subfolder name within the provided base_dir
    dest_dir = base_dir / "Small-ImageNet-Validation-Dataset-1000-Classes"

    dest_dir.parent.mkdir(parents=True, exist_ok=True)

    git_dir = dest_dir / ".git"

    if dest_dir.exists():
        # If it exists and is a git repo, optionally update
        if git_dir.exists() and update_if_exists:
            try:
                subprocess.check_call(["git", "-C", str(dest_dir), "fetch", "--all", "--prune"])  # nosec
                subprocess.check_call(["git", "-C", str(dest_dir), "pull", "--ff-only"])  # nosec
            except subprocess.CalledProcessError as e:
                print(f"Warning: failed to update existing dataset repo: {e}")
            except FileNotFoundError:
                print("Warning: 'git' not found on PATH; skipping update of existing dataset repo.")
        # Either updated or non-git directory with data — return as-is
        # If present, prefer returning the images subdirectory if available
        images_dir = dest_dir / "ILSVRC2012_img_val_subset"
        return images_dir if images_dir.exists() else dest_dir

    # Clone fresh
    try:
        subprocess.check_call(  # nosec
            [
                "git",
                "clone",
                "--depth",
                "1",
                repo_url,
                str(dest_dir),
            ]
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "'git' executable not found. Please install git to clone the sample images."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Failed to clone dataset repository: {exc}") from exc

    # Quick sanity check for expected files
    expected_json = dest_dir / "imagenet_class_index.json"
    expected_images_dir = dest_dir / "ILSVRC2012_img_val_subset"
    if not expected_json.exists() or not expected_images_dir.exists():
        print(
            "Warning: cloned repository is missing expected files. "
            "Ensure 'imagenet_class_index.json' and 'ILSVRC2012_img_val_subset' are present."
        )

    # Return the images subdirectory containing class folders
    return expected_images_dir if expected_images_dir.exists() else dest_dir


class ImageNetSampleDataset(Dataset):
    """Dataset for ImageNet sample images organized in class directories."""

    def __init__(self, image_dir: str, transform=None, target_size: int = 256):
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.target_size = target_size

        # Find all image files organized by class directories
        self.image_files: List[Path] = []
        self.class_indices: List[int] = []

        # Iterate through class directories (named 0, 1, 2, ..., 999)
        for class_dir in sorted(self.image_dir.iterdir()):
            if class_dir.is_dir() and class_dir.name.isdigit():
                class_idx = int(class_dir.name)

                # Find all image files in this class directory
                image_files_in_class: List[Path] = []
                for ext in ["*.JPEG", "*.jpg", "*.png", "*.JPG"]:
                    image_files_in_class.extend(list(class_dir.glob(ext)))

                # Add all images from this class
                for img_file in image_files_in_class:
                    self.image_files.append(img_file)
                    self.class_indices.append(class_idx)

        print(
            f"Found {len(self.image_files)} valid images across {len(set(self.class_indices))} classes"
        )

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, index: int):
        img_path = self.image_files[index]
        class_idx = self.class_indices[index]

        # Load image
        with Image.open(img_path).convert("RGB") as img:
            image = img.copy()

        # Apply transform if provided
        if self.transform is not None:
            image = self.transform(image)

        filename = img_path.name
        return image, class_idx, filename


def load_demo_models(device: str = "cuda") -> Dict[str, torch.nn.Module]:
    """Load the three XCiT model variants for comparison."""
    models = {}

    model_configs = [
        ("xcit_nano_12_p16_pretrained", "Baseline XCiT"),
        ("xcit_aps_nano_12_p16_pretrained", "APS XCiT"),
        ("xcit_af_small_12_p16_pretrained", "AF XCiT"),
    ]

    print("Loading models...")
    for model_name, description in model_configs:
        try:
            model = create_model(model_name, pretrained=False, num_classes=1000)
            model.to(device)
            model.eval()

            # Ensure all components are on the correct device
            for param in model.parameters():
                param.data = param.data.to(device)
            for buffer in model.buffers():
                buffer.data = buffer.data.to(device)

            models[model_name] = model
            print(f"Loaded {description}")
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")

    return models


def create_data_loader(
    repo_dir: Path,
    input_size: int = 256,
    batch_size: int = 4,
    limit: Optional[int] = None,
) -> DataLoader:
    """Create data loader for the sample images."""
    transform = transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    base_dataset = ImageNetSampleDataset(
        image_dir=repo_dir, transform=transform, target_size=input_size
    )
    if limit is not None and limit > 0:
        limit = min(limit, len(base_dataset))
        dataset = Subset(base_dataset, list(range(limit)))
    else:
        dataset = base_dataset

    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def apply_crop_shift(
    images: torch.Tensor, shift_x: int, shift_y: int, crop_size: int
) -> torch.Tensor:
    """Apply crop shift to images."""
    _, _, height, width = images.shape

    # Calculate crop boundaries
    center_x, center_y = width // 2, height // 2
    start_x = center_x + shift_x - crop_size // 2
    start_y = center_y + shift_y - crop_size // 2

    # Clamp to valid range
    start_x = max(0, min(start_x, width - crop_size))
    start_y = max(0, min(start_y, height - crop_size))

    # Crop
    cropped = images[:, :, start_y:start_y + crop_size, start_x:start_x + crop_size]
    return cropped


def evaluate_crop_shift_robustness(
    models: Dict[str, torch.nn.Module],
    data_loader: DataLoader,
    max_shift: int = 4,
    crop_size: int = 224,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Evaluate crop shift robustness for all models."""
    results = {"per_sample_results": [], "af_robust_cases": [], "summary_stats": {}}

    print(
        f"Evaluating crop shift robustness (max_shift={max_shift}, crop_size={crop_size})"
    )

    for _, (images, targets, filenames) in enumerate(
        tqdm(data_loader, desc="Processing batches")
    ):
        images = images.to(device)
        targets = targets.to(device)
        batch_size = images.size(0)

        # Initialize per-sample results
        batch_results = []
        for i in range(batch_size):
            sample_result = {
                "filename": filenames[i],
                "target": targets[i].item(),
                "shifts_correct": {name: [] for name in models.keys()},
                "max_robust_shift": {name: -1 for name in models.keys()},
            }
            batch_results.append(sample_result)

        # Test different shift levels
        for shift in range(max_shift + 1):
            if shift == 0:
                shift_positions = [(0, 0)]  # Center crop
            else:
                # Test shifts in cardinal and diagonal directions
                shift_positions = []
                for dx in [-shift, 0, shift]:
                    for dy in [-shift, 0, shift]:
                        if abs(dx) == shift or abs(dy) == shift:
                            shift_positions.append((dx, dy))

            # Track correctness for this shift level
            shift_correct = {
                name: torch.ones(batch_size, dtype=torch.bool, device=device)
                for name in models.keys()
            }

            for shift_x, shift_y in shift_positions:
                shifted_images = apply_crop_shift(images, shift_x, shift_y, crop_size)

                # Evaluate all models
                for model_name, model in models.items():
                    with torch.no_grad():
                        outputs = model(shifted_images)
                        predictions = torch.argmax(outputs, dim=1)
                        correct = predictions == targets
                        shift_correct[model_name] &= correct

            # Update per-sample results
            for i, sample_result in enumerate(batch_results):
                for model_name in models.keys():
                    is_correct = shift_correct[model_name][i].item()
                    sample_result["shifts_correct"][model_name].append(is_correct)

                    if is_correct:
                        sample_result["max_robust_shift"][model_name] = shift

        # Check for AF robust cases matching strict GIF criteria
        for i, sample_result in enumerate(batch_results):
            af_robust = sample_result["max_robust_shift"].get(
                "xcit_af_small_12_p16_pretrained", -1
            )
            baseline_robust = sample_result["max_robust_shift"].get(
                "xcit_nano_12_p16_pretrained", -1
            )
            aps_robust = sample_result["max_robust_shift"].get(
                "xcit_aps_nano_12_p16_pretrained", -1
            )
            # Only include cases where:
            # 1) All models are correct at center (robust >= 0)
            # 2) Baseline and APS are NOT perfectly robust (< max_shift)
            # 3) AF IS perfectly robust (== max_shift)
            all_center_correct = (
                baseline_robust >= 0 and aps_robust >= 0 and af_robust >= 0
            )
            baseline_not_perfect = baseline_robust < max_shift
            aps_not_perfect = aps_robust < max_shift
            af_perfect = af_robust == max_shift

            if (
                all_center_correct
                and baseline_not_perfect
                and aps_not_perfect
                and af_perfect
            ):
                results["af_robust_cases"].append(
                    {
                        "filename": sample_result["filename"],
                        "target": sample_result["target"],
                        # For directory-organized datasets, the class directory equals the target index
                        "class_idx": sample_result["target"],
                        "af_robust_shift": af_robust,
                        "baseline_robust_shift": baseline_robust,
                        "aps_robust_shift": aps_robust,
                    }
                )

        results["per_sample_results"].extend(batch_results)

    # Compute summary statistics
    results["summary_stats"] = compute_summary_stats(
        results["per_sample_results"], max_shift
    )

    return results


def build_fractional_grid(
    dx: float,
    dy: float,
    H: int,
    W: int,
    device,
    align_corners: bool = True,
) -> torch.Tensor:
    """Build grid for fractional shifts using bilinear interpolation."""
    iy = torch.arange(1, H - 1, device=device) + dy
    ix = torch.arange(1, W - 1, device=device) + dx

    if align_corners:
        gy = -1.0 + 2.0 * iy / (H - 1)
        gx = -1.0 + 2.0 * ix / (W - 1)
    else:
        gy = -1.0 + 2.0 * (iy + 0.5) / H
        gx = -1.0 + 2.0 * (ix + 0.5) / W

    gy, gx = torch.meshgrid(gy, gx, indexing="ij")
    return torch.stack((gx, gy), dim=-1).unsqueeze(0)


def evaluate_fractional_shift_robustness(
    models: Dict[str, torch.nn.Module],
    data_loader: DataLoader,
    max_shift: int = 4,
    device: str = "cuda",
    align_corners: bool = True,
) -> Dict[str, Any]:
    """Evaluate fractional shift robustness using bilinear interpolation."""
    results = {"per_sample_results": [], "af_robust_cases": [], "summary_stats": {}}

    fractions = [k / max_shift for k in range(-max_shift, max_shift + 1)]

    print(f"Evaluating fractional shift robustness (max_shift={max_shift} steps)")

    for _, (images, targets, filenames) in enumerate(
        tqdm(data_loader, desc="Processing batches")
    ):
        # Pad images for fractional shifts (need border for interpolation)
        padded_images = F.pad(images, (1, 1, 1, 1), mode="reflect").to(device)
        targets = targets.to(device)
        batch_size = images.size(0)
        H, W = padded_images.shape[2], padded_images.shape[3]

        # Pre-build grids for this batch
        grid_bank = {
            (dx, dy): build_fractional_grid(dx, dy, H, W, device, align_corners)
            for dx in fractions
            for dy in fractions
        }

        # Initialize per-sample results
        batch_results = []
        for i in range(batch_size):
            sample_result = {
                "filename": filenames[i],
                "target": targets[i].item(),
                "shifts_correct": {name: [] for name in models.keys()},
                "max_robust_shift": {name: -1 for name in models.keys()},
            }
            batch_results.append(sample_result)

        # Baseline evaluation (center crop)
        center_images = padded_images[:, :, 1:-1, 1:-1]
        baseline_correct = {}
        for model_name, model in models.items():
            with torch.no_grad():
                outputs = model(center_images)
                predictions = torch.argmax(outputs, dim=1)
                baseline_correct[model_name] = predictions == targets

        # Update baseline results
        for i, sample_result in enumerate(batch_results):
            for model_name in models.keys():
                is_correct = baseline_correct[model_name][i].item()
                sample_result["shifts_correct"][model_name].append(is_correct)
                if is_correct:
                    sample_result["max_robust_shift"][model_name] = 0

        # Initialize robust flags
        robust_flags = {name: baseline_correct[name].clone() for name in models.keys()}

        # Test fractional shifts
        for s in range(1, max_shift + 1):
            frac = s / max_shift
            shifts_lvl = [
                (dx, dy)
                for dx in fractions
                for dy in fractions
                if max(abs(dx), abs(dy)) == frac
            ]

            for dx, dy in shifts_lvl:
                grid = grid_bank[(dx, dy)].repeat(batch_size, 1, 1, 1)
                shifted_images = F.grid_sample(
                    padded_images, grid, mode="bilinear", align_corners=align_corners
                )

                for model_name, model in models.items():
                    with torch.no_grad():
                        outputs = model(shifted_images)
                        predictions = torch.argmax(outputs, dim=1)
                        correct = predictions == targets
                        robust_flags[model_name] &= correct

            # Update results for this shift level
            for i, sample_result in enumerate(batch_results):
                for model_name in models.keys():
                    is_correct = robust_flags[model_name][i].item()
                    if is_correct:
                        sample_result["max_robust_shift"][model_name] = s

        # Check for AF robust cases matching strict GIF criteria
        for i, sample_result in enumerate(batch_results):
            af_robust = sample_result["max_robust_shift"].get(
                "xcit_af_small_12_p16_pretrained", -1
            )
            baseline_robust = sample_result["max_robust_shift"].get(
                "xcit_nano_12_p16_pretrained", -1
            )
            aps_robust = sample_result["max_robust_shift"].get(
                "xcit_aps_nano_12_p16_pretrained", -1
            )
            # Only include cases where:
            # 1) All models are correct at center (robust >= 0)
            # 2) Baseline and APS are NOT perfectly robust (< max_shift)
            # 3) AF IS perfectly robust (== max_shift)
            all_center_correct = (
                baseline_robust >= 0 and aps_robust >= 0 and af_robust >= 0
            )
            baseline_not_perfect = baseline_robust < max_shift
            aps_not_perfect = aps_robust < max_shift
            af_perfect = af_robust == max_shift

            if (
                all_center_correct
                and baseline_not_perfect
                and aps_not_perfect
                and af_perfect
            ):
                results["af_robust_cases"].append(
                    {
                        "filename": sample_result["filename"],
                        "target": sample_result["target"],
                        # For directory-organized datasets, the class directory equals the target index
                        "class_idx": sample_result["target"],
                        "af_robust_shift": af_robust,
                        "baseline_robust_shift": baseline_robust,
                        "aps_robust_shift": aps_robust,
                    }
                )

        results["per_sample_results"].extend(batch_results)

    # Compute summary statistics
    results["summary_stats"] = compute_summary_stats(
        results["per_sample_results"], max_shift
    )

    return results


def compute_summary_stats(
    per_sample_results: List[Dict], max_shift: int
) -> Dict[str, Any]:
    """Compute summary statistics from per-sample results."""
    stats = {}

    if not per_sample_results:
        return stats

    model_names = list(per_sample_results[0]["max_robust_shift"].keys())

    for model_name in model_names:
        robust_shifts = [
            result["max_robust_shift"][model_name] for result in per_sample_results
        ]

        # Basic statistics
        stats[model_name] = {
            "mean_robust_shift": np.mean(robust_shifts),
            "median_robust_shift": np.median(robust_shifts),
            "std_robust_shift": np.std(robust_shifts),
            "robust_at_shift": {},
        }

        # Robustness at different shift levels
        for shift in range(max_shift + 1):
            robust_count = sum(1 for rs in robust_shifts if rs >= shift)
            stats[model_name]["robust_at_shift"][shift] = robust_count / len(
                robust_shifts
            )

    return stats


def create_visualization_gif(
    models: Dict[str, torch.nn.Module],
    image_path: str,
    target_class: int,
    max_shift: int,
    shift_type: str = "crop",
    output_path: str = "demo_visualization.gif",
    device: str = "cuda",
) -> None:
    """Create a GIF showing shift robustness for a specific image."""

    # Load and preprocess image
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image = Image.open(image_path).convert("RGB")
    tensor_image = transform(image).unsqueeze(0).to(device)

    # Prepare for visualization
    frames = []

    if shift_type == "crop":
        # Pad image for crop shifts
        padded_image = F.pad(
            tensor_image, (max_shift, max_shift, max_shift, max_shift), mode="reflect"
        )

        for shift in range(max_shift + 1):
            if shift == 0:
                shifts = [(0, 0)]
            else:
                # Match evaluator: include all positions on the square perimeter at this radius
                shifts = []
                for dx in range(-shift, shift + 1):
                    for dy in range(-shift, shift + 1):
                        if abs(dx) == shift or abs(dy) == shift:
                            shifts.append((dx, dy))

            for shift_x, shift_y in shifts:
                cropped = apply_crop_shift(padded_image, shift_x, shift_y, 224)

                # Get predictions
                predictions = {}
                for model_name, model in models.items():
                    with torch.no_grad():
                        output = model(cropped)
                        pred = torch.argmax(output, dim=1).item()
                        predictions[model_name] = pred

                # Create frame
                frame = create_prediction_frame(
                    cropped, predictions, target_class, shift_x, shift_y
                )
                frames.append(frame)

    elif shift_type == "fractional":
        # Pad for fractional shifts
        padded_image = F.pad(tensor_image, (1, 1, 1, 1), mode="reflect")
        H, W = padded_image.shape[2], padded_image.shape[3]

        for s in range(max_shift + 1):
            if s == 0:
                shifts = [(0.0, 0.0)]
            else:
                frac = s / max_shift
                shifts = [(frac, 0), (-frac, 0), (0, frac), (0, -frac)]

            for dx, dy in shifts:
                grid = build_fractional_grid(dx, dy, H, W, device)
                shifted = F.grid_sample(
                    padded_image, grid, mode="bilinear", align_corners=True
                )

                # Get predictions
                predictions = {}
                for model_name, model in models.items():
                    with torch.no_grad():
                        output = model(shifted)
                        pred = torch.argmax(output, dim=1).item()
                        predictions[model_name] = pred

                # Create frame
                frame = create_prediction_frame(
                    shifted, predictions, target_class, dx, dy
                )
                frames.append(frame)

    # Save GIF
    imageio.mimsave(output_path, frames, duration=1.0, loop=0, fps=5)
    print(f"Visualization saved to {output_path}")


def create_prediction_frame(
    image_tensor: torch.Tensor,
    predictions: Dict[str, int],
    target_class: int,
    shift_x: float,
    shift_y: float,
) -> np.ndarray:
    """Create a publication-ready frame with clean typography and alignment."""

    # Style
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
        }
    )

    # Denormalize image for display
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = image_tensor.squeeze(0).cpu() * std + mean
    img = torch.clamp(img, 0, 1)

    # Figure layout
    fig = plt.figure(figsize=(8, 5), dpi=100)
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 2])
    ax_img = fig.add_subplot(gs[0, 0])
    ax_txt = fig.add_subplot(gs[0, 1])

    # Image panel
    ax_img.imshow(img.permute(1, 2, 0))
    # ax_img.set_title(f"Shift: ({shift_x:.2f}, {shift_y:.2f})", pad=8)
    ax_img.set_title(f"Shift: ({shift_x:.2f}, {shift_y:.2f})")
    ax_img.axis("off")

    # Text panel
    ax_txt.axis("off")
    ax_txt.set_xlim(0, 1)
    ax_txt.set_ylim(0, 1)

    # Target class name
    target_name_raw = (
        IMAGENET_CLASSES[target_class]
        if target_class < len(IMAGENET_CLASSES)
        else f"class_{target_class}"
    )
    target_name = prettify_imagenet_label(target_name_raw)

    ax_txt.text(
        0.02,
        0.92,
        "Target:",
        fontsize=13,
        fontweight="bold",
        color="#111111",
        ha="left",
        va="top",
    )
    ax_txt.text(
        0.44,
        0.92,
        f"{target_name}",
        fontsize=13,
        fontweight="bold",
        color="#111111",
        ha="left",
        va="top",
    )


    # Predictions with clear green/red coloring
    model_names = {
        "xcit_nano_12_p16_pretrained": "Baseline",
        "xcit_aps_nano_12_p16_pretrained": "APS",
        "xcit_af_small_12_p16_pretrained": "AF",
    }
    y = 0.80
    line_gap = 0.20
    for model_name, pred in predictions.items():
        short_name = model_names.get(model_name, model_name)
        is_correct = pred == target_class
        color = "#2ca02c" if is_correct else "#d62728"
        pred_name_raw = IMAGENET_CLASSES[pred] if pred < len(IMAGENET_CLASSES) else f"class_{pred}"
        pred_name = prettify_imagenet_label(pred_name_raw)
        # Left-aligned model name in black, right-aligned prediction in color
        ax_txt.text(
            0.02,
            y,
            f"{short_name}:",
            fontsize=12,
            fontweight="bold",
            color="#111111",
            ha="left",
            va="center",
        )
        ax_txt.text(
            0.4, 
            y,
            pred_name,
            fontsize=12,
            color=color,
            ha="left",
            va="center",
        )
        y -= line_gap

    # Tight margins
    # Minimize outer margins to avoid extra white space and prevent title clipping
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.02, wspace=0.10)

    # To numpy
    fig.canvas.draw()
    frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return frame


def create_gifs_for_af_cases(
    models: Dict[str, torch.nn.Module],
    af_robust_cases: List[Dict],
    repo_dir: Path,
    max_shift: int,
    shift_type: str = "crop",
    output_dir: str = "af_gifs",
    device: str = "cuda",
) -> List[str]:
    """Create GIFs for all cases where AF model is more robust."""

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    gif_files = []

    print(f"Creating {len(af_robust_cases)} GIFs for AF-robust cases...")

    for i, case in enumerate(tqdm(af_robust_cases, desc="Creating GIFs")):
        # Images are stored under class index directories (e.g., repo_dir/123/imagename.JPEG)
        class_idx = case.get("class_idx", case["target"])
        image_path = repo_dir / str(class_idx) / case["filename"]
        target_class = case["target"]

        # Create descriptive filename
        clean_filename = Path(case["filename"]).stem
        gif_filename = f"{shift_type}_{i+1:03d}_c{class_idx}_{clean_filename}.gif"
        gif_path = output_path / gif_filename

        try:
            create_visualization_gif(
                models=models,
                image_path=str(image_path),
                target_class=target_class,
                max_shift=max_shift,
                shift_type=shift_type,
                output_path=str(gif_path),
                device=device,
            )
            gif_files.append(str(gif_path))

        except Exception as e:
            print(f"Error creating GIF for {case['filename']}: {e}")
            continue

    print(f"Created {len(gif_files)} GIFs in {output_path}")
    return gif_files


def print_summary_results(results: Dict[str, Any], shift_type: str = "crop"):
    """Print a summary of the robustness evaluation results."""
    print(f"\n{'='*60}")
    print(f"{shift_type.upper()} SHIFT ROBUSTNESS SUMMARY")
    print(f"{'='*60}")

    summary_stats = results["summary_stats"]
    model_names = {
        "xcit_nano_12_p16_pretrained": "Baseline XCiT",
        "xcit_aps_nano_12_p16_pretrained": "APS XCiT",
        "xcit_af_small_12_p16_pretrained": "AF XCiT",
    }

    for model_name, stats in summary_stats.items():
        display_name = model_names.get(model_name, model_name)
        print(f"\n{display_name}:")
        print(f"  Mean robust shift: {stats['mean_robust_shift']:.2f}")
        print(f"  Median robust shift: {stats['median_robust_shift']:.2f}")

        robust_at_shift = stats.get("robust_at_shift", {})
        for shift_level in sorted(robust_at_shift.keys()):
            if shift_level > 0:
                print(
                    f"  Robust at shift {shift_level}: {robust_at_shift[shift_level]*100:.1f}%"
                )

    af_cases = results["af_robust_cases"]
    print(f"\nAF model is more robust than baseline/APS on {len(af_cases)} images")

    if af_cases:
        print("\nExamples where AF model excels:")
        for i, case in enumerate(af_cases[:3]):  # Show first 3 examples
            print(
                f"  {i+1}. {case['filename']}: AF={case['af_robust_shift']}, "
                f"Baseline={case['baseline_robust_shift']}, APS={case['aps_robust_shift']}"
            )
