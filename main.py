
import os
import re
import json
import csv
import traceback
from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk
import matplotlib.pyplot as plt
import trimesh

from PIL import Image
from skimage.measure import marching_cubes


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

# Automatically detect the folder containing main.py.
PROJECT = Path(__file__).resolve().parent

DATASET = PROJECT / "dataset"
OUTPUT = PROJECT / "outputs"

VOLUME_DIR = OUTPUT / "reconstructed_volumes"
MASK_DIR = OUTPUT / "segmentation_masks"
MESH_DIR = OUTPUT / "3d_models"
PREVIEW_DIR = OUTPUT / "previews"
REPORT_DIR = OUTPUT / "quality_reports"
BLENDER_DIR = OUTPUT / "blender_scenes"

for folder in [
    VOLUME_DIR,
    MASK_DIR,
    MESH_DIR,
    PREVIEW_DIR,
    REPORT_DIR,
    BLENDER_DIR
]:
    folder.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. GENERAL HELPERS
# ============================================================

def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(value))


def natural_key(path):
    numbers = re.findall(r"\d+", Path(path).name)
    return int(numbers[-1]) if numbers else 0


def find_chaos_root(split_folder):
    """Find the directory containing CT and MR patient folders."""

    if not split_folder.exists():
        return None

    for folder in [split_folder] + list(split_folder.rglob("*")):

        if not folder.is_dir():
            continue

        names = {
            p.name.lower()
            for p in folder.iterdir()
            if p.is_dir()
        }

        if "ct" in names and ("mr" in names or "mri" in names):
            return folder

    return None


def patient_folders(modality_folder):

    if not modality_folder.exists():
        return []

    folders = [
        p for p in modality_folder.iterdir()
        if p.is_dir()
    ]

    numeric = [
        p for p in folders
        if p.name.isdigit()
    ]

    return sorted(numeric or folders, key=lambda p: p.name)


def get_dicom_files(folder):

    if not folder.exists():
        return []

    return sorted([
        p for p in folder.rglob("*")
        if p.is_file()
        and p.suffix.lower() in [".dcm", ".dicom"]
    ])


# ============================================================
# 3. READ DICOM FILES AND RECONSTRUCT A 3D VOLUME
# ============================================================

def read_dicom_series(files):

    slices = []
    errors = []

    for filename in files:

        try:
            ds = pydicom.dcmread(
                str(filename),
                force=True
            )

            if not hasattr(ds, "PixelData"):
                continue

            pixel = ds.pixel_array.astype(np.float32)

            slope = float(getattr(ds, "RescaleSlope", 1))
            intercept = float(getattr(ds, "RescaleIntercept", 0))

            pixel = pixel * slope + intercept

            orientation = np.asarray(
                getattr(
                    ds,
                    "ImageOrientationPatient",
                    [1, 0, 0, 0, 1, 0]
                ),
                dtype=float
            )

            position = getattr(
                ds,
                "ImagePositionPatient",
                None
            )

            if position is not None:
                position = np.asarray(position, dtype=float)

            slices.append({
                "pixel": pixel,
                "position": position,
                "orientation": orientation,
                "spacing": np.asarray(
                    getattr(ds, "PixelSpacing", [1, 1]),
                    dtype=float
                ),
                "instance": int(
                    getattr(ds, "InstanceNumber", len(slices))
                ),
                "thickness": float(
                    getattr(
                        ds,
                        "SpacingBetweenSlices",
                        getattr(ds, "SliceThickness", 1)
                    )
                )
            })

        except Exception as error:
            errors.append(f"{filename.name}: {error}")

    if not slices:
        raise ValueError("No readable DICOM image slices found.")

    first = slices[0]

    iop = first["orientation"]

    row_direction = iop[:3]
    column_direction = iop[3:]

    normal = np.cross(row_direction, column_direction)
    normal = normal / np.linalg.norm(normal)

    # Sort slices using physical position.
    if all(item["position"] is not None for item in slices):

        slices.sort(
            key=lambda item: float(
                np.dot(item["position"], normal)
            )
        )

        positions = np.array([
            np.dot(item["position"], normal)
            for item in slices
        ])

        gaps = np.diff(positions)

        if len(gaps):
            z_spacing = float(np.median(gaps))
        else:
            z_spacing = first["thickness"]

    else:

        slices.sort(
            key=lambda item: item["instance"]
        )

        z_spacing = first["thickness"]
        gaps = np.array([])

    # Check slice dimensions.
    shapes = {
        item["pixel"].shape
        for item in slices
    }

    if len(shapes) != 1:
        raise ValueError(
            "DICOM slices have inconsistent image dimensions."
        )

    array = np.stack([
        item["pixel"]
        for item in slices
    ]).astype(np.float32)

    row_spacing, column_spacing = first["spacing"]

    spacing = (
        float(column_spacing),
        float(row_spacing),
        float(z_spacing)
    )

    if all(item["position"] is not None for item in slices):
        origin = tuple(
            float(v)
            for v in slices[0]["position"]
        )
    else:
        origin = (0.0, 0.0, 0.0)

    direction_matrix = np.column_stack([
        row_direction,
        column_direction,
        normal
    ])

    image = sitk.GetImageFromArray(array)

    image.SetSpacing(spacing)
    image.SetOrigin(origin)

    image.SetDirection(
        tuple(direction_matrix.flatten())
    )

    issues = []

    if errors:
        issues.append(
            f"{len(errors)} DICOM files could not be read."
        )

    if len(gaps) > 1 and z_spacing > 0:

        if np.max(gaps) > 1.5 * z_spacing:
            issues.append(
                "Possible missing slice or large slice gap detected."
            )

        if np.std(gaps) / z_spacing > 0.05:
            issues.append(
                "Slice spacing is irregular; review before using the volume."
            )

    for item in slices:

        if not np.allclose(
            item["orientation"],
            iop,
            atol=1e-4
        ):
            issues.append(
                "Inconsistent slice orientation detected."
            )
            break

    metadata = {
        "slice_count": len(slices),
        "dimensions": list(image.GetSize()),
        "spacing_mm": list(image.GetSpacing()),
        "origin_mm": list(image.GetOrigin()),
        "direction": list(image.GetDirection()),
        "geometry_issues": sorted(set(issues)),
        "unreadable_files": errors
    }

    return image, array, metadata


# ============================================================
# 4. MRI ECHO TIME GROUPING
# ============================================================

def group_mri_files(dicom_folder):

    files = get_dicom_files(dicom_folder)

    groups = {}

    for filename in files:

        try:
            ds = pydicom.dcmread(
                str(filename),
                stop_before_pixels=True,
                force=True
            )

            echo = getattr(ds, "EchoTime", None)

            parent = str(filename.parent).lower()

            if "inphase" in parent:
                key = "InPhase"

            elif "opphase" in parent or "outphase" in parent:
                key = "OutPhase"

            elif echo is not None:
                key = f"Echo_{float(echo):.3f}"

            else:
                key = "UnknownEcho"

            groups.setdefault(key, []).append(filename)

        except Exception:
            continue

    return groups


def process_mri_sequence(
    sequence_folder,
    patient_id,
    split
):

    dicom_folder = sequence_folder / "DICOM_anon"

    groups = group_mri_files(dicom_folder)

    if not groups:
        print("No MRI DICOM files found:", dicom_folder)
        return []

    group_items = list(groups.items())

    if sequence_folder.name.upper() == "T1DUAL":

        def echo_sort(item):

            if item[0].startswith("Echo_"):
                return float(item[0].split("_")[1])

            if item[0] == "OutPhase":
                return -1

            if item[0] == "InPhase":
                return 1

            return 999

        group_items.sort(key=echo_sort)

        if len(group_items) == 2:

            names = ["OutPhase", "InPhase"]

            group_items = [
                (names[i], item[1])
                for i, item in enumerate(group_items)
            ]

    results = []

    for phase, files in group_items:

        if phase == "UnknownEcho":
            phase = "MRI"

        series_name = safe_name(
            f"{split}_MR_{patient_id}_"
            f"{sequence_folder.name}_{phase}"
        )

        result = process_one_series(
            files=files,
            series_name=series_name,
            modality="MRI",
            split=split,
            ground_folder=sequence_folder / "Ground",
            reverse_mask=False
        )

        if result:
            results.append(result)

    return results


# ============================================================
# 5. LOAD CHAOS GROUND-TRUTH MASKS
# ============================================================

def read_png_masks(ground_folder):

    if not ground_folder.exists():
        return None

    files = sorted(
        list(ground_folder.glob("*.png")),
        key=natural_key
    )

    if not files:
        return None

    arrays = []

    for filename in files:

        image = Image.open(filename).convert("L")

        arrays.append(
            np.asarray(image)
        )

    return np.stack(arrays)


def convert_mask_labels(raw_mask, modality):

    if modality == "CT":
        # CHAOS CT liver mask.
        return (raw_mask > 0).astype(np.uint8)

    # CHAOS MRI grayscale labels:
    # 1 = liver
    # 2 = right kidney
    # 3 = left kidney
    # 4 = spleen

    labels = np.zeros(
        raw_mask.shape,
        dtype=np.uint8
    )

    labels[
        (raw_mask > 55) & (raw_mask <= 70)
    ] = 1

    labels[
        (raw_mask > 110) & (raw_mask <= 135)
    ] = 2

    labels[
        (raw_mask > 175) & (raw_mask <= 200)
    ] = 3

    labels[
        (raw_mask > 240) & (raw_mask <= 255)
    ] = 4

    return labels


# ============================================================
# 6. GENERATE PHYSICAL-COORDINATE 3D ORGAN MESHES
# ============================================================

ORGAN_NAMES = {
    1: "Liver",
    2: "Right_Kidney",
    3: "Left_Kidney",
    4: "Spleen"
}


def generate_mesh(mask, image, organ_name, filename):

    if np.count_nonzero(mask) < 10:
        return None

    try:

        vertices_zyx, faces, normals, values = marching_cubes(
            mask.astype(np.float32),
            level=0.5
        )

        # Convert Z,Y,X to X,Y,Z.
        indices_xyz = vertices_zyx[:, [2, 1, 0]]

        spacing = np.asarray(
            image.GetSpacing()
        )

        origin = np.asarray(
            image.GetOrigin()
        )

        direction = np.asarray(
            image.GetDirection()
        ).reshape(3, 3)

        physical_lps = (
            origin +
            (direction @ (indices_xyz * spacing).T).T
        )

        # Convert LPS to RAS/Blender coordinates.
        physical_blender = physical_lps * np.array([
            -1.0, -1.0, 1.0
        ])

        mesh = trimesh.Trimesh(
            vertices=physical_blender,
            faces=faces,
            process=True
        )

        if mesh.is_empty:
            return None

        mesh.export(str(filename))

        return {
            "organ": organ_name,
            "mesh_file": str(filename),
            "vertices": int(len(mesh.vertices)),
            "faces": int(len(mesh.faces)),
            "watertight": bool(mesh.is_watertight),
            "mesh_volume_mm3": float(abs(mesh.volume))
        }

    except Exception as error:

        print(
            "Mesh generation failed:",
            organ_name,
            error
        )

        return None


# ============================================================
# 7. SAVE PREVIEW AND QUALITY REPORT
# ============================================================

def save_preview(array, name):

    middle = array.shape[0] // 2

    low, high = np.percentile(
        array[middle],
        [1, 99]
    )

    if high <= low:

        low = float(np.min(array[middle]))
        high = float(np.max(array[middle])) + 1

    plt.figure(figsize=(7, 7))

    plt.imshow(
        array[middle],
        cmap="gray",
        vmin=low,
        vmax=high
    )

    plt.title(name)
    plt.axis("off")
    plt.tight_layout()

    filename = PREVIEW_DIR / f"{name}.png"

    plt.savefig(
        filename,
        dpi=150
    )

    plt.close()

    return filename


# ============================================================
# 8. PROCESS ONE COMPLETE IMAGE SERIES
# ============================================================

def process_one_series(
    files,
    series_name,
    modality,
    split,
    ground_folder=None,
    reverse_mask=False
):

    try:

        print("\nProcessing:", series_name)

        image, array, quality = read_dicom_series(files)

        # Save reconstructed 3D volume.
        volume_path = VOLUME_DIR / f"{series_name}.nii.gz"

        sitk.WriteImage(
            image,
            str(volume_path)
        )

        preview_path = save_preview(
            array,
            series_name
        )

        quality["modality"] = modality
        quality["split"] = split
        quality["series_name"] = series_name
        quality["intensity_min"] = float(np.min(array))
        quality["intensity_max"] = float(np.max(array))
        quality["intensity_mean"] = float(np.mean(array))
        quality["volume_file"] = str(volume_path)
        quality["preview_file"] = str(preview_path)

        mesh_results = []

        raw_mask = read_png_masks(ground_folder)

        if raw_mask is not None:

            if reverse_mask:
                raw_mask = raw_mask[::-1]

            if raw_mask.shape != array.shape:

                quality["mask_status"] = (
                    "SKIPPED: mask and DICOM dimensions/order "
                    "do not match. Check slice alignment."
                )

            else:

                labels = convert_mask_labels(
                    raw_mask,
                    modality
                )

                mask_image = sitk.GetImageFromArray(labels)

                mask_image.CopyInformation(image)

                mask_path = (
                    MASK_DIR /
                    f"{series_name}_mask.nii.gz"
                )

                sitk.WriteImage(
                    mask_image,
                    str(mask_path)
                )

                quality["mask_status"] = "MATCHED"
                quality["mask_file"] = str(mask_path)

                voxel_volume = float(
                    np.prod(image.GetSpacing())
                )

                for label, organ in ORGAN_NAMES.items():

                    if modality == "CT" and organ != "Liver":
                        continue

                    organ_mask = (
                        labels == label
                    ).astype(np.uint8)

                    voxel_count = int(
                        np.count_nonzero(organ_mask)
                    )

                    if voxel_count == 0:
                        continue

                    mesh_name = safe_name(
                        f"{series_name}_{organ}"
                    )

                    mesh_path = (
                        MESH_DIR /
                        f"{mesh_name}.stl"
                    )

                    mesh_result = generate_mesh(
                        organ_mask,
                        image,
                        organ,
                        mesh_path
                    )

                    if mesh_result:

                        mesh_result["voxel_count"] = voxel_count

                        mesh_result["voxel_volume_cm3"] = round(
                            voxel_count * voxel_volume / 1000,
                            3
                        )

                        mesh_results.append(mesh_result)

        else:

            quality["mask_status"] = (
                "No reference mask found. "
                "Automatic organ segmentation is not performed."
            )

        quality["mesh_results"] = mesh_results

        quality["overall_geometry_status"] = (
            "WARNING"
            if quality["geometry_issues"]
            else "PASS"
        )

        report_path = (
            REPORT_DIR /
            f"{series_name}.json"
        )

        with open(
            report_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                quality,
                file,
                indent=4
            )

        print("Modality:", modality)
        print("Volume dimensions:", array.shape)
        print("Physical spacing:", image.GetSpacing())
        print(
            "Geometry:",
            quality["overall_geometry_status"]
        )
        print("Mask:", quality["mask_status"])
        print("3D organ meshes:", len(mesh_results))

        return {
            "series": series_name,
            "modality": modality,
            "split": split,
            "slices": len(files),
            "geometry_status": quality["overall_geometry_status"],
            "mask_status": quality["mask_status"],
            "mesh_count": len(mesh_results),
            "volume": str(volume_path),
            "report": str(report_path)
        }

    except Exception as error:

        print("ERROR:", series_name, error)

        traceback.print_exc()

        return None


# ============================================================
# 9. PROCESS CT DATA
# ============================================================

def process_ct_patient(patient, split):

    dicom_folder = patient / "DICOM_anon"

    files = get_dicom_files(dicom_folder)

    if not files:

        print("No CT DICOM files:", patient)

        return []

    series_name = safe_name(
        f"{split}_CT_Patient_{patient.name}"
    )

    result = process_one_series(
        files=files,
        series_name=series_name,
        modality="CT",
        split=split,
        ground_folder=patient / "Ground",
        reverse_mask=True
    )

    return [result] if result else []


# ============================================================
# 10. PROCESS MRI DATA
# ============================================================

def process_mri_patient(patient, split):

    results = []

    for sequence in ["T1DUAL", "T2SPIR"]:

        sequence_folder = patient / sequence

        if not sequence_folder.exists():
            continue

        results.extend(
            process_mri_sequence(
                sequence_folder,
                patient.name,
                split
            )
        )

    return results


# ============================================================
# 11. GENERATE BLENDER IMPORT SCRIPT
# ============================================================

def create_blender_import_script():

    blender_script = f'''
import bpy
from pathlib import Path

MESH_DIR = Path({str(MESH_DIR)!r})
SAVE_DIR = Path({str(BLENDER_DIR)!r})

SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Clear default objects.
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)

bpy.context.scene.unit_settings.system = "METRIC"
bpy.context.scene.unit_settings.scale_length = 0.001

COLORS = {{
    "Liver": (0.65, 0.12, 0.10, 1.0),
    "Right_Kidney": (0.90, 0.30, 0.20, 1.0),
    "Left_Kidney": (0.20, 0.45, 0.90, 1.0),
    "Spleen": (0.65, 0.25, 0.65, 1.0)
}}

def create_material(name, color):

    material = bpy.data.materials.new(name=name)
    material.diffuse_color = color
    material.use_nodes = True

    bsdf = material.node_tree.nodes.get("Principled BSDF")

    if bsdf:
        bsdf.inputs["Base Color"].default_value = color

    return material

mesh_files = list(MESH_DIR.glob("*.stl"))

for file in mesh_files:

    try:
        if hasattr(bpy.ops.wm, "stl_import"):
            bpy.ops.wm.stl_import(filepath=str(file))
        else:
            bpy.ops.import_mesh.stl(filepath=str(file))

        obj = bpy.context.object
        obj.name = file.stem

        for organ, color in COLORS.items():

            if organ.lower() in file.stem.lower():

                material = create_material(organ, color)
                obj.data.materials.append(material)
                break

        bpy.ops.object.shade_smooth()

        print("Imported:", file.name)

    except Exception as error:
        print("Import error:", file, error)

save_file = SAVE_DIR / "CHAOS_Anatomical_Models.blend"

bpy.ops.wm.save_as_mainfile(filepath=str(save_file))

print("Blender scene saved:", save_file)
'''

    script_path = (
        BLENDER_DIR /
        "blender_import_scene.py"
    )

    with open(
        script_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(blender_script)

    return script_path


# ============================================================
# 12. MAIN PROGRAM
# ============================================================

def main():

    print("\n========================================")
    print("PATIENT-SPECIFIC 3D RECONSTRUCTION")
    print("CHAOS CT + MRI PIPELINE")
    print("========================================")

    print("Project folder:", PROJECT)
    print("Dataset folder:", DATASET)
    print("Output folder:", OUTPUT)

    if not DATASET.exists():

        print("\nERROR: Dataset folder does not exist:")
        print(DATASET)

        return

    all_results = []

    for split_name in ["Training", "Testing"]:

        split_folder = DATASET / split_name

        if not split_folder.exists():

            print("Split folder missing:", split_folder)

            continue

        root = find_chaos_root(split_folder)

        if root is None:

            print(
                "Could not locate CT and MR folders in:",
                split_folder
            )

            continue

        print("\nDATASET:", split_name)
        print("CHAOS root:", root)

        ct_root = root / "CT"

        mr_root = root / "MR"

        if not mr_root.exists():
            mr_root = root / "MRI"

        # Process CT patients.
        for patient in patient_folders(ct_root):

            all_results.extend(
                process_ct_patient(
                    patient,
                    split_name
                )
            )

        # Process MRI patients.
        for patient in patient_folders(mr_root):

            all_results.extend(
                process_mri_patient(
                    patient,
                    split_name
                )
            )

    # Save overall CSV report.
    summary_file = (
        OUTPUT /
        "reconstruction_summary.csv"
    )

    columns = [
        "series",
        "modality",
        "split",
        "slices",
        "geometry_status",
        "mask_status",
        "mesh_count",
        "volume",
        "report"
    ]

    with open(
        summary_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=columns
        )

        writer.writeheader()
        writer.writerows(all_results)

    # Generate Blender import script.
    blender_script = create_blender_import_script()

    print("\n========================================")
    print("PIPELINE FINISHED")
    print("========================================")

    print(
        "Total reconstructed series:",
        len(all_results)
    )

    print(
        "CT series:",
        sum(
            x["modality"] == "CT"
            for x in all_results
        )
    )

    print(
        "MRI series:",
        sum(
            x["modality"] == "MRI"
            for x in all_results
        )
    )

    print(
        "Generated organ meshes:",
        sum(
            x["mesh_count"]
            for x in all_results
        )
    )

    print("\nOUTPUT FOLDERS:")

    print("Volumes:", VOLUME_DIR)
    print("Masks:", MASK_DIR)
    print("3D models:", MESH_DIR)
    print("Previews:", PREVIEW_DIR)
    print("Reports:", REPORT_DIR)
    print("Blender script:", blender_script)
    print("Summary:", summary_file)


# ============================================================
# 13. PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
