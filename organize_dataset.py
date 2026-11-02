
from pathlib import Path
import zipfile

# Main project folder
PROJECT = Path(
    r"C:\Users\DELL\OneDrive\Desktop\Patient_3D_Reconstruction"
)

# Folder containing your downloaded ZIP files
DOWNLOADS = Path.home() / "Downloads"

DATASET = PROJECT / "dataset"

# Create project folders
for folder in [
    PROJECT / "outputs",
    PROJECT / "src",
    DATASET / "Training",
    DATASET / "Testing"
]:
    folder.mkdir(parents=True, exist_ok=True)

# Find ZIP files
zip_files = list(DOWNLOADS.glob("*.zip"))

print("ZIP files found:")
for i, file in enumerate(zip_files):
    print(i, file.name)

# Select the ZIP files by their names
for zip_file in zip_files:
    name = zip_file.name.lower()

    if "train" in name:
        destination = DATASET / "Training"
    elif "test" in name:
        destination = DATASET / "Testing"
    else:
        continue

    # Avoid extracting again
    if any(destination.iterdir()):
        print(f"Already contains files: {destination}")
        continue

    print(f"\nExtracting: {zip_file.name}")

    with zipfile.ZipFile(zip_file, "r") as archive:
        archive.extractall(destination)

    print(f"Extracted into: {destination}")

print("\nProject folder setup completed!")
