
import os
import json
import glob
import csv
import traceback
from datetime import datetime

import numpy as np
import trimesh

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak
)

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

QUALITY_DIR = os.path.join(
    BASE_DIR, "outputs", "quality_reports"
)

PREVIEW_DIR = os.path.join(
    BASE_DIR, "outputs", "previews"
)

MODEL_DIR = os.path.join(
    BASE_DIR, "outputs", "3d_models"
)

REPORT_DIR = os.path.join(
    BASE_DIR, "outputs", "final_reports"
)

INDIVIDUAL_DIR = os.path.join(
    REPORT_DIR, "individual_reports"
)

DATA_DIR = os.path.join(
    REPORT_DIR, "report_data"
)

os.makedirs(INDIVIDUAL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# --------------------------------------------------
# PDF STYLES
# --------------------------------------------------

styles = getSampleStyleSheet()

styles.add(ParagraphStyle(
    name="MainTitle",
    parent=styles["Title"],
    fontSize=19,
    leading=24,
    textColor=colors.HexColor("#17365D"),
    spaceAfter=12
))

styles.add(ParagraphStyle(
    name="SectionTitle",
    parent=styles["Heading2"],
    fontSize=13,
    textColor=colors.HexColor("#176B87"),
    spaceBefore=12,
    spaceAfter=7
))

styles.add(ParagraphStyle(
    name="SmallText",
    parent=styles["BodyText"],
    fontSize=8,
    leading=11
))


def safe_text(value):
    """Convert values into safe PDF text."""
    if value is None:
        return "Not available"

    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# --------------------------------------------------
# FIND RELATED FILES
# --------------------------------------------------

def find_preview(series_name):

    possible = [
        os.path.join(PREVIEW_DIR, series_name + ".png"),
        os.path.join(PREVIEW_DIR, series_name + ".jpg"),
        os.path.join(PREVIEW_DIR, series_name + ".jpeg")
    ]

    for path in possible:
        if os.path.isfile(path):
            return path

    return None


def find_models(series_name):

    patterns = [
        os.path.join(MODEL_DIR, series_name + "*.stl"),
        os.path.join(MODEL_DIR, series_name + "*.obj"),
        os.path.join(MODEL_DIR, series_name + "*.ply")
    ]

    found = []

    for pattern in patterns:
        found.extend(glob.glob(pattern))

    return sorted(set(found))


# --------------------------------------------------
# MESH QUALITY INFORMATION
# --------------------------------------------------

def get_mesh_information(model_path):

    result = {
        "filename": os.path.basename(model_path),
        "vertices": "Unavailable",
        "faces": "Unavailable",
        "watertight": "Unavailable",
        "components": "Unavailable",
        "surface_area": "Unavailable",
        "volume": "Not calculated"
    }

    try:
        mesh = trimesh.load(
            model_path,
            force="mesh",
            process=False
        )

        if mesh.is_empty:
            return result

        result["vertices"] = len(mesh.vertices)
        result["faces"] = len(mesh.faces)
        result["watertight"] = bool(mesh.is_watertight)

        components = mesh.split(only_watertight=False)
        result["components"] = len(components)

        result["surface_area"] = round(
            float(mesh.area), 3
        )

        # STL units are not guaranteed.
        # Only report volume for watertight meshes.
        if mesh.is_watertight:
            result["volume"] = round(
                abs(float(mesh.volume)), 3
            )

    except Exception as error:
        result["error"] = str(error)

    return result


# --------------------------------------------------
# COMMON PDF TABLE
# --------------------------------------------------

def make_table(rows):

    table = Table(
        rows,
        colWidths=[65 * mm, 105 * mm],
        repeatRows=1
    )

    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0),
         colors.HexColor("#17365D")),

        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),

        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),

        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),

        ("FONTSIZE", (0, 0), (-1, -1), 8),

        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),

        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#EEF3F7")]),

        ("LEFTPADDING", (0, 0), (-1, -1), 6),

        ("RIGHTPADDING", (0, 0), (-1, -1), 6),

        ("TOPPADDING", (0, 0), (-1, -1), 6),

        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))

    return table


# --------------------------------------------------
# INDIVIDUAL PATIENT REPORT
# --------------------------------------------------

def generate_patient_report(json_path):

    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    series_name = data.get(
        "series_name",
        os.path.splitext(os.path.basename(json_path))[0]
    )

    output_path = os.path.join(
        INDIVIDUAL_DIR,
        series_name + "_Report.pdf"
    )

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm
    )

    story = []

    # TITLE
    story.append(Paragraph(
        "PATIENT-SPECIFIC 3D RECONSTRUCTION REPORT",
        styles["MainTitle"]
    ))

    story.append(Paragraph(
        "DICOM Geometry, Segmentation and Mesh Assessment",
        styles["Heading3"]
    ))

    story.append(Spacer(1, 8))

    story.append(Paragraph(
        "Generated on: " + datetime.now().strftime(
            "%d-%m-%Y %H:%M:%S"
        ),
        styles["SmallText"]
    ))

    story.append(Spacer(1, 12))

    # CASE INFORMATION
    story.append(Paragraph(
        "1. Case Information",
        styles["SectionTitle"]
    ))

    case_rows = [
        ["Parameter", "Value"],
        ["Case / Series", safe_text(series_name)],
        ["Dataset split", safe_text(data.get("split"))],
        ["Modality", safe_text(data.get("modality"))],
        ["Slice count", safe_text(data.get("slice_count"))],
        ["Dimensions", safe_text(data.get("dimensions"))]
    ]

    story.append(make_table(case_rows))

    # DICOM GEOMETRY
    story.append(Paragraph(
        "2. DICOM Geometry Information",
        styles["SectionTitle"]
    ))

    geometry_rows = [
        ["Parameter", "Value"],
        ["Spacing (mm)", safe_text(data.get("spacing_mm"))],
        ["Origin (mm)", safe_text(data.get("origin_mm"))],
        ["Direction", safe_text(data.get("direction"))],
        ["Geometry status",
         safe_text(data.get("overall_geometry_status"))],
        ["Geometry issues",
         safe_text(data.get("geometry_issues"))],
        ["Unreadable files",
         safe_text(data.get("unreadable_files"))]
    ]

    story.append(make_table(geometry_rows))

    # INTENSITY INFORMATION
    story.append(Paragraph(
        "3. Image Intensity Information",
        styles["SectionTitle"]
    ))

    intensity_rows = [
        ["Parameter", "Value"],
        ["Minimum intensity", safe_text(data.get("intensity_min"))],
        ["Maximum intensity", safe_text(data.get("intensity_max"))],
        ["Mean intensity", safe_text(data.get("intensity_mean"))]
    ]

    story.append(make_table(intensity_rows))

    # SEGMENTATION
    story.append(Paragraph(
        "4. Segmentation Status",
        styles["SectionTitle"]
    ))

    story.append(Paragraph(
        safe_text(data.get(
            "mask_status",
            "Segmentation status not recorded"
        )),
        styles["BodyText"]
    ))

    # PREVIEW IMAGE
    preview_path = find_preview(series_name)

    story.append(Paragraph(
        "5. Image Preview",
        styles["SectionTitle"]
    ))

    if preview_path:

        try:
            preview = Image(
                preview_path,
                width=150 * mm,
                height=105 * mm,
                kind="proportional"
            )

            story.append(preview)

        except Exception:
            story.append(Paragraph(
                "Preview image could not be inserted.",
                styles["BodyText"]
            ))

    else:

        story.append(Paragraph(
            "No preview image found.",
            styles["BodyText"]
        ))

    # MESH INFORMATION
    story.append(Paragraph(
        "6. 3D Mesh Assessment",
        styles["SectionTitle"]
    ))

    model_files = find_models(series_name)

    if model_files:

        mesh_rows = [
            [
                "Model",
                "Vertices",
                "Faces",
                "Watertight",
                "Components"
            ]
        ]

        for model_path in model_files:

            info = get_mesh_information(model_path)

            mesh_rows.append([
                safe_text(info["filename"]),
                str(info["vertices"]),
                str(info["faces"]),
                str(info["watertight"]),
                str(info["components"])
            ])

        mesh_table = Table(
            mesh_rows,
            colWidths=[60 * mm, 25 * mm, 25 * mm,
                       28 * mm, 30 * mm],
            repeatRows=1
        )

        mesh_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0),
             colors.HexColor("#17365D")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("WORDWRAP", (0, 0), (-1, -1), True)
        ]))

        story.append(mesh_table)

        # Save machine-readable mesh metrics
        mesh_json_path = os.path.join(
            DATA_DIR,
            series_name + "_mesh_metrics.json"
        )

        mesh_info = [
            get_mesh_information(path)
            for path in model_files
        ]

        with open(mesh_json_path, "w", encoding="utf-8") as f:
            json.dump(mesh_info, f, indent=4)

    else:

        story.append(Paragraph(
            "No corresponding 3D mesh files were found.",
            styles["BodyText"]
        ))

    # LIMITATIONS
    story.append(Paragraph(
        "7. Validation Notes and Limitations",
        styles["SectionTitle"]
    ))

    story.append(Paragraph(
        "A geometry PASS indicates that the available geometry "
        "checks did not detect an issue. It does not independently "
        "establish anatomical accuracy. Segmentation accuracy "
        "requires a correctly aligned reference mask. Mesh "
        "measurements depend on the coordinate system and unit "
        "scale of the exported model. This report is intended for "
        "research and visualization, not clinical diagnosis.",
        styles["SmallText"]
    ))

    doc.build(story)

    return output_path


# --------------------------------------------------
# COMBINED DATASET REPORT
# --------------------------------------------------

def generate_combined_report(all_data):

    output_path = os.path.join(
        REPORT_DIR,
        "combined_dataset_report.pdf"
    )

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4
    )

    story = []

    story.append(Paragraph(
        "COMBINED DATASET RECONSTRUCTION REPORT",
        styles["MainTitle"]
    ))

    story.append(Paragraph(
        "CHAOS CT and MRI Dataset",
        styles["Heading2"]
    ))

    story.append(Spacer(1, 15))

    total = len(all_data)

    geometry_pass = sum(
        1 for item in all_data
        if item.get("overall_geometry_status") == "PASS"
    )

    ct_count = sum(
        1 for item in all_data
        if str(item.get("modality", "")).upper() == "CT"
    )

    mr_count = sum(
        1 for item in all_data
        if str(item.get("modality", "")).upper() in ["MR", "MRI"]
    )

    training_count = sum(
        1 for item in all_data
        if str(item.get("split", "")).lower() == "training"
    )

    testing_count = sum(
        1 for item in all_data
        if str(item.get("split", "")).lower() == "testing"
    )

    rows = [
        ["Metric", "Result"],
        ["Total processed series", str(total)],
        ["Geometry PASS", str(geometry_pass)],
        ["CT series", str(ct_count)],
        ["MRI series", str(mr_count)],
        ["Training series", str(training_count)],
        ["Testing series", str(testing_count)]
    ]

    story.append(make_table(rows))

    story.append(Paragraph(
        "Case-wise Summary",
        styles["SectionTitle"]
    ))

    summary_rows = [
        ["Series", "Modality", "Split", "Geometry", "Mask status"]
    ]

    for item in all_data:

        summary_rows.append([
            safe_text(item.get("series_name", "Unknown")),
            safe_text(item.get("modality")),
            safe_text(item.get("split")),
            safe_text(item.get("overall_geometry_status")),
            safe_text(item.get("mask_status", "Not available"))
        ])

    summary_table = Table(
        summary_rows,
        colWidths=[48 * mm, 20 * mm, 22 * mm,
                   25 * mm, 55 * mm],
        repeatRows=1
    )

    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0),
         colors.HexColor("#17365D")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 6.5),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("WORDWRAP", (0, 0), (-1, -1), True)
    ]))

    story.append(summary_table)

    story.append(Spacer(1, 15))

    story.append(Paragraph(
        "Interpretation: Geometry PASS represents the result "
        "of the implemented geometry checks only. It is not "
        "a measure of segmentation accuracy or clinical "
        "performance. Testing cases without reference masks "
        "are not assigned Dice or IoU scores.",
        styles["SmallText"]
    ))

    doc.build(story)

    return output_path


# --------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------

def main():

    print("\n======================================")
    print(" AUTOMATIC REPORT GENERATION")
    print("======================================")

    json_files = glob.glob(
        os.path.join(QUALITY_DIR, "**", "*.json"),
        recursive=True
    )

    if not json_files:
        print("No JSON quality reports found.")
        print("Check outputs/quality_reports/")
        return

    all_data = []
    success_count = 0

    for json_path in json_files:

        try:

            print("\nProcessing:", os.path.basename(json_path))

            with open(json_path, "r", encoding="utf-8") as file:
                data = json.load(file)

            all_data.append(data)

            pdf_path = generate_patient_report(json_path)

            print("PDF created:", pdf_path)

            success_count += 1

        except Exception as error:

            print("Error processing:", json_path)
            print(error)
            traceback.print_exc()

    if all_data:

        try:

            combined_path = generate_combined_report(all_data)

            print("\nCombined report created:")
            print(combined_path)

        except Exception as error:

            print("Combined report error:", error)
            traceback.print_exc()

    # Export a simple CSV summary
    csv_path = os.path.join(
        REPORT_DIR,
        "dataset_summary.csv"
    )

    fields = [
        "series_name",
        "modality",
        "split",
        "slice_count",
        "dimensions",
        "spacing_mm",
        "overall_geometry_status",
        "mask_status"
    ]

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields,
            extrasaction="ignore"
        )

        writer.writeheader()

        for item in all_data:

            writer.writerow({
                **item,
                "dimensions": str(item.get("dimensions")),
                "spacing_mm": str(item.get("spacing_mm"))
            })

    print("\n======================================")
    print(" REPORT GENERATION COMPLETED")
    print("======================================")

    print("Successful individual reports:", success_count)
    print("Total JSON reports:", len(json_files))
    print("Output folder:", REPORT_DIR)


if __name__ == "__main__":
    main()
