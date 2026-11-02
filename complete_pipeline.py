
"""
PATIENT-SPECIFIC 3D ANATOMICAL RECONSTRUCTION
CHAOS DATASET | CT + MRI | BLENDER | MCP

Replaces previous generated files for the selected split/modality.
Does not delete dataset, src, main.py or unrelated project files.

Note:
This pipeline reconstructs meshes from available Ground masks.
It does not predict missing masks or train a segmentation model.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import trimesh
from PIL import Image
from scipy import ndimage
from skimage.measure import marching_cubes

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "dataset"
OUTPUT = ROOT / "outputs"

VOLUME_DIR = OUTPUT / "reconstructed_volumes"
MASK_DIR = OUTPUT / "segmentation_masks"
MESH_DIR = OUTPUT / "3d_models"
PREVIEW_DIR = OUTPUT / "previews"
REPORT_DIR = OUTPUT / "quality_reports"
BLENDER_DIR = OUTPUT / "blender_scenes"

OUTPUT_DIRS = [
    VOLUME_DIR,
    MASK_DIR,
    MESH_DIR,
    PREVIEW_DIR,
    REPORT_DIR,
    BLENDER_DIR,
]

MRI_LABELS = {
    1: "Liver",
    2: "Right_Kidney",
    3: "Left_Kidney",
    4: "Spleen",
}

COLORS = {
    "Liver": (0.75, 0.12, 0.10, 1.0),
    "Right_Kidney": (0.85, 0.35, 0.18, 1.0),
    "Left_Kidney": (0.95, 0.55, 0.20, 1.0),
    "Spleen": (0.55, 0.18, 0.45, 1.0),
}


# ============================================================
# 2. FILE MANAGEMENT
# ============================================================

def safe_name(value):
    return re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        str(value)
    ).strip("_")


def natural_key(value):
    return [
        int(x) if x.isdigit() else x.lower()
        for x in re.split(r"(\d+)", str(value))
    ]


def prepare_output_folders():
    for folder in OUTPUT_DIRS:
        folder.mkdir(parents=True, exist_ok=True)


def replace_previous_outputs(split, modality):
    """
    Remove only generated files belonging to the selected
    dataset split and modality.

    Example:
        Training_CT_*.stl
        Training_CT_*.nii.gz
        Training_CT_*.png
        Training_CT_*.json

    Other modalities and unrelated files are preserved.
    """
    prefix = f"{split}_{modality}_"

    print("\n[REPLACE] Checking previous output files...")
    print("[REPLACE] Selected prefix:", prefix)

    for folder in OUTPUT_DIRS:
        folder.mkdir(parents=True, exist_ok=True)

        for file in folder.rglob("*"):
            if file.is_file() and file.name.startswith(prefix):
                try:
                    file.unlink()
                    print("[REPLACED]", file.name)
                except PermissionError:
                    raise PermissionError(
                        f"Cannot replace {file}. Close any application "
                        "using this file and try again."
                    )

    print("[REPLACE] Previous matching outputs cleared.\n")


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, default=str)


# ============================================================
# 3. DICOM DISCOVERY
# ============================================================

def find_series(folder):
    folder = Path(folder)

    try:
        ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(
            str(folder)
        )
    except Exception:
        return []

    if not ids:
        return []

    result = []

    for series_id in ids:
        files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(
            str(folder), series_id
        )

        if files:
            result.append({
                "id": series_id,
                "files": list(files),
            })

    return result


def discover_dicom_folders(root):
    found = []

    for current, dirs, files in os.walk(root):
        dirs[:] = [
            d for d in dirs
            if d.lower() not in {
                "ground", "outputs", "__pycache__"
            }
        ]

        if files and find_series(current):
            found.append(Path(current))

    return sorted(set(found), key=natural_key)


def read_volume(series):
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(series["files"])

    image = reader.Execute()
    array = sitk.GetArrayFromImage(image)

    if array.ndim != 3:
        raise ValueError("DICOM series is not a 3D volume.")

    return image, array


# ============================================================
# 4. PREPROCESSING AND PREVIEWS
# ============================================================

def normalize_volume(array, modality):
    array = array.astype(np.float32)

    if modality == "CT":
        array = np.clip(array, -200, 300)
    else:
        low, high = np.percentile(array, [1, 99])

        if high > low:
            array = np.clip(array, low, high)

    low = np.min(array)
    high = np.max(array)

    if high > low:
        array = (array - low) / (high - low)

    return array


def save_preview(array, path, title, cmap="gray"):
    middle = array.shape[0] // 2

    plt.figure(figsize=(7, 7))
    plt.imshow(array[middle], cmap=cmap)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# 5. GROUND MASK DISCOVERY
# ============================================================

def find_ground_folder(dicom_folder):
    folder = Path(dicom_folder)

    candidates = [
        folder.parent / "Ground",
        folder.parent.parent / "Ground",
        folder.parent.parent.parent / "Ground",
    ]

    for candidate in candidates:
        if candidate.is_dir():
            return candidate

    return None


def read_mask_stack(files):
    files = sorted(files, key=natural_key)

    slices = []

    for file in files:
        image = Image.open(file).convert("L")
        slices.append(np.asarray(image))

    if not slices:
        return None

    if len({x.shape for x in slices}) != 1:
        return None

    return np.stack(slices, axis=0)


def find_matching_mask_stacks(ground, volume_shape):
    groups = {}

    for file in ground.rglob("*.png"):
        groups.setdefault(file.parent, []).append(file)

    matching = []

    for folder, files in groups.items():
        stack = read_mask_stack(files)

        if stack is not None and stack.shape == tuple(volume_shape):
            matching.append((folder, stack))

    return matching


# ============================================================
# 6. MASK CLEANING
# ============================================================

def clean_mask(mask):
    mask = mask.astype(bool)

    mask = ndimage.binary_fill_holes(mask)
    mask = ndimage.binary_opening(mask, iterations=1)

    labels, count = ndimage.label(mask)

    if count:
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        largest = np.argmax(sizes)
        mask = labels == largest

    return mask.astype(np.uint8)


# ============================================================
# 7. 3D MESH GENERATION
# ============================================================

def generate_mesh(mask, image, mesh_path):
    mask = clean_mask(mask)

    if np.count_nonzero(mask) < 30:
        raise ValueError("Not enough foreground voxels.")

    spacing_xyz = image.GetSpacing()
    spacing_zyx = (
        spacing_xyz[2],
        spacing_xyz[1],
        spacing_xyz[0]
    )

    vertices_zyx, faces, _, _ = marching_cubes(
        mask,
        level=0.5,
        spacing=spacing_zyx
    )

    # Convert Z,Y,X to X,Y,Z.
    vertices_xyz = vertices_zyx[:, ::-1]

    origin = np.asarray(image.GetOrigin())
    spacing = np.asarray(spacing_xyz)
    direction = np.asarray(
        image.GetDirection()
    ).reshape(3, 3)

    physical_lps = (
        origin + (vertices_xyz * spacing) @ direction.T
    )

    # Convert LPS coordinates to RAS coordinates.
    physical_ras = physical_lps.copy()
    physical_ras[:, 0] *= -1
    physical_ras[:, 1] *= -1

    mesh = trimesh.Trimesh(
        vertices=physical_ras,
        faces=faces,
        process=True
    )

    # Smooth mesh surface.
    try:
        trimesh.smoothing.filter_taubin(
            mesh,
            lamb=0.5,
            nu=0.53,
            iterations=8
        )
    except Exception as error:
        print("[WARNING] Smoothing skipped:", error)

    mesh.remove_unreferenced_vertices()
    mesh.export(str(mesh_path))

    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
        "surface_area": float(mesh.area),
        "volume": float(abs(mesh.volume)),
        "bounds": mesh.bounds.tolist(),
    }


# ============================================================
# 8. BLENDER SCENE
# ============================================================

def create_blender_script(mesh_files, scene_path, script_path):
    mesh_data = [
        {
            "path": str(p.resolve()),
            "name": p.stem
        }
        for p in mesh_files
    ]

    script = r'''
import bpy
import json
from mathutils import Vector

mesh_data = json.loads(r"""__DATA__""")
scene_path = r"""__SCENE__"""

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)

colors = {
    "Liver": (0.75, 0.12, 0.10, 1.0),
    "Right_Kidney": (0.85, 0.35, 0.18, 1.0),
    "Left_Kidney": (0.95, 0.55, 0.20, 1.0),
    "Spleen": (0.55, 0.18, 0.45, 1.0)
}

for item in mesh_data:
    try:
        bpy.ops.wm.stl_import(filepath=item["path"])
    except Exception:
        bpy.ops.import_mesh.stl(filepath=item["path"])

    obj = bpy.context.selected_objects[0]
    obj.name = item["name"]

    organ = next(
        (name for name in colors if name in item["name"]),
        None
    )

    color = colors.get(
        organ, (0.2, 0.65, 0.75, 1.0)
    )

    material = bpy.data.materials.new(
        item["name"] + "_Material"
    )
    material.diffuse_color = color
    material.use_nodes = True

    material.node_tree.nodes["Principled BSDF"].inputs[
        "Base Color"
    ].default_value = color

    obj.data.materials.clear()
    obj.data.materials.append(material)

    for polygon in obj.data.polygons:
        polygon.use_smooth = True

bpy.ops.object.camera_add(location=(350, -450, 300))
camera = bpy.context.object

direction = Vector((0, 0, 0)) - camera.location
camera.rotation_euler = direction.to_track_quat(
    "-Z", "Y"
).to_euler()

bpy.context.scene.camera = camera

bpy.ops.object.light_add(
    type="AREA",
    location=(0, -250, 350)
)

light = bpy.context.object
light.data.energy = 1800
light.data.size = 250

bpy.ops.wm.save_as_mainfile(filepath=scene_path)
print("Scene saved:", scene_path)
'''

    script = script.replace(
        "__DATA__", json.dumps(mesh_data)
    ).replace(
        "__SCENE__", str(scene_path.resolve())
    )

    script_path.write_text(script, encoding="utf-8")


def open_blender(mesh_files, blender_path=None):
    if not mesh_files:
        print("[INFO] No STL files available.")
        return

    blender_path = blender_path or shutil.which("blender")

    if not blender_path:
        print("[WARNING] Blender was not found in PATH.")
        print("You can import STL files manually in Blender.")
        return

    scene_path = BLENDER_DIR / "Patient_Reconstruction.blend"
    script_path = BLENDER_DIR / "build_scene.py"

    create_blender_script(
        mesh_files,
        scene_path,
        script_path
    )

    result = subprocess.run(
        [
            str(blender_path),
            "--background",
            "--python",
            str(script_path)
        ],
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        print("[SUCCESS] Blender scene:", scene_path)
    else:
        print("[ERROR] Blender scene generation failed.")
        print(result.stderr[-3000:])


# ============================================================
# 9. PROCESS ONE DICOM SERIES
# ============================================================

def process_one_series(
    dicom_folder,
    series,
    split,
    modality,
    series_number
):
    patient = dicom_folder.parent.name
    folder_name = dicom_folder.name

    name = safe_name(
        f"{split}_{modality}_{patient}_{folder_name}_"
        f"Series{series_number}"
    )

    print("\n[PROCESSING]", name)

    image, volume = read_volume(series)

    print("[INFO] Volume shape:", volume.shape)
    print("[INFO] Spacing:", image.GetSpacing())

    volume_path = VOLUME_DIR / f"{name}.nii.gz"
    sitk.WriteImage(image, str(volume_path))

    normalized = normalize_volume(volume, modality)

    preview_path = PREVIEW_DIR / f"{name}_volume.png"

    save_preview(
        normalized,
        preview_path,
        f"{modality} volume: {name}"
    )

    report = {
        "name": name,
        "split": split,
        "modality": modality,
        "dicom_folder": str(dicom_folder),
        "volume_shape_zyx": list(volume.shape),
        "spacing_xyz": list(image.GetSpacing()),
        "origin": list(image.GetOrigin()),
        "volume_file": str(volume_path),
        "preview_file": str(preview_path),
        "mask_stacks_found": 0,
        "meshes": [],
        "warnings": []
    }

    ground = find_ground_folder(dicom_folder)

    if ground is None:
        report["warnings"].append(
            "No Ground folder found."
        )
    else:
        candidates = find_matching_mask_stacks(
            ground,
            volume.shape
        )

        report["mask_stacks_found"] = len(candidates)

        if not candidates:
            report["warnings"].append(
                "No mask stack has the same dimensions as this volume."
            )

        for stack_index, (mask_folder, stack) in enumerate(
            candidates, start=1
        ):
            labels = [
                int(x) for x in np.unique(stack)
                if int(x) != 0
            ]

            for label in labels:
                if modality == "CT":
                    organ = "Liver"
                else:
                    organ = MRI_LABELS.get(
                        label, f"Class_{label}"
                    )

                mask = (stack == label).astype(np.uint8)

                if np.count_nonzero(mask) < 30:
                    continue

                base = safe_name(
                    f"{name}_{organ}_Stack{stack_index}"
                )

                mask_path = MASK_DIR / f"{base}.nii.gz"
                mesh_path = MESH_DIR / f"{base}.stl"
                mask_preview = PREVIEW_DIR / f"{base}_mask.png"

                mask_image = sitk.GetImageFromArray(mask)

                # This copies image geometry provisionally.
                # Matching dimensions do not establish true alignment.
                mask_image.CopyInformation(image)

                sitk.WriteImage(mask_image, str(mask_path))

                save_preview(
                    mask,
                    mask_preview,
                    f"{organ} mask",
                    cmap="nipy_spectral"
                )

                try:
                    mesh_info = generate_mesh(
                        mask,
                        image,
                        mesh_path
                    )

                    report["meshes"].append({
                        "organ": organ,
                        "mask": str(mask_path),
                        "mesh": str(mesh_path),
                        "preview": str(mask_preview),
                        **mesh_info
                    })

                    print("[SUCCESS] Created:", mesh_path.name)

                except Exception as error:
                    report["warnings"].append(
                        f"{organ} mesh failed: {error}"
                    )

    report["warnings"].append(
        "Verify Ground mask slice ordering, orientation and physical "
        "alignment before using the mesh for measurements or evaluation."
    )

    report_path = REPORT_DIR / f"{name}.json"
    save_json(report_path, report)

    return report


# ============================================================
# 10. PROCESS TRAINING OR TESTING
# ============================================================

def process_dataset(
    split="Training",
    modality="CT",
    replace=True
):
    split = split.upper()
    modality = modality.upper()

    if split not in {"TRAINING", "TESTING"}:
        raise ValueError("Split must be Training or Testing.")

    if modality not in {"CT", "MR", "BOTH"}:
        raise ValueError("Modality must be CT, MR or BOTH.")

    split_folder = DATASET / (
        "Training" if split == "TRAINING" else "Testing"
    )

    if not split_folder.is_dir():
        raise FileNotFoundError(
            f"Dataset folder not found: {split_folder}"
        )

    modalities = ["CT", "MR"] if modality == "BOTH" else [modality]

    all_reports = []

    for current_modality in modalities:

        if replace:
            replace_previous_outputs(
                "Training" if split == "TRAINING" else "Testing",
                current_modality
            )

        modality_folder = split_folder / current_modality

        if not modality_folder.is_dir():
            print("[WARNING] Missing:", modality_folder)
            continue

        dicom_folders = discover_dicom_folders(
            modality_folder
        )

        print(
            f"\n[INFO] {current_modality}: "
            f"{len(dicom_folders)} DICOM folders found."
        )

        for dicom_folder in dicom_folders:
            series_list = find_series(dicom_folder)

            for number, series in enumerate(
                series_list, start=1
            ):
                try:
                    result = process_one_series(
                        dicom_folder,
                        series,
                        "Training" if split == "TRAINING" else "Testing",
                        current_modality,
                        number
                    )
                    all_reports.append(result)

                except Exception as error:
                    print("[ERROR]", dicom_folder, error)

    summary = {
        "split": split,
        "modality": modality,
        "series_processed": len(all_reports),
        "mesh_count": sum(
            len(x["meshes"]) for x in all_reports
        ),
        "results": all_reports
    }

    summary_name = (
        f"{'Training' if split == 'TRAINING' else 'Testing'}_"
        f"{modality}_summary.json"
    )

    summary_path = REPORT_DIR / summary_name
    save_json(summary_path, summary)

    print("\n========================================")
    print("PROCESSING COMPLETED")
    print("Series:", summary["series_processed"])
    print("Meshes:", summary["mesh_count"])
    print("Report:", summary_path)
    print("========================================")

    return summary


# ============================================================
# 11. LIST GENERATED OUTPUTS
# ============================================================

def list_outputs():
    result = {}

    for folder in OUTPUT_DIRS:
        result[folder.name] = [
            str(file)
            for file in folder.rglob("*")
            if file.is_file()
        ]

    return result


# ============================================================
# 12. OPTIONAL MCP SERVER
# ============================================================

def start_mcp():
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as error:
        raise RuntimeError(
            "MCP SDK import failed. Check the MCP installation."
        ) from error

    server = FastMCP("Patient-3D-Reconstruction")

    @server.tool()
    def run_reconstruction(
        split: str = "Training",
        modality: str = "CT"
    ) -> str:
        """Run the CHAOS reconstruction pipeline."""
        result = process_dataset(split, modality)

        return json.dumps({
            "series_processed": result["series_processed"],
            "mesh_count": result["mesh_count"]
        })

    @server.tool()
    def show_outputs() -> str:
        """List generated project files."""
        return json.dumps(list_outputs(), indent=2)

    server.run()


# ============================================================
# 13. MAIN PROGRAM
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Patient-specific 3D reconstruction pipeline"
    )

    parser.add_argument(
        "--process",
        choices=["Training", "Testing"],
        default="Training"
    )

    parser.add_argument(
        "--modality",
        choices=["CT", "MR", "BOTH"],
        default="CT"
    )

    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace previous outputs for the selected modality."
    )

    parser.add_argument(
        "--open-blender",
        action="store_true"
    )

    parser.add_argument(
        "--blender",
        default=None,
        help="Full path to blender.exe if not in PATH."
    )

    parser.add_argument(
        "--list-outputs",
        action="store_true"
    )

    parser.add_argument(
        "--mcp",
        action="store_true"
    )

    args = parser.parse_args()

    prepare_output_folders()

    if args.list_outputs:
        print(json.dumps(list_outputs(), indent=2))
        return

    if args.mcp:
        start_mcp()
        return

    result = process_dataset(
        args.process,
        args.modality,
        replace=args.replace
    )

    if args.open_blender:
        mesh_files = [
            Path(mesh["mesh"])
            for report in result["results"]
            for mesh in report["meshes"]
        ]

        open_blender(
            mesh_files,
            args.blender
        )


if __name__ == "__main__":
    main()
