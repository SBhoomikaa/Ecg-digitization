import json
import cv2
from pathlib import Path

# Load the ECG image
ecg_image_path = "C:\\Users\\008bh\\Downloads\\ptb-xl\\ptbxl\\records_mod\\00004_lr-0_0000.png"
json_file_path = "C:\\Users\\008bh\\Downloads\\ptb-xl\\ptbxl\\records_mod\\00004_lr-0_0000.json"
output_folder = "D:\\cropped_leads"

image = cv2.imread(ecg_image_path)
with open(json_file_path, 'r') as f:
    data = json.load(f)

Path(output_folder).mkdir(parents=True, exist_ok=True)

for lead in data['leads']:
    if 'lead_name' not in lead:
        continue

    lead_name = lead['lead_name']
    bbox = lead['lead_bounding_box']

    # Coordinates are [y, x] not [x, y]!
    points = [bbox[str(i)] for i in range(4)]
    y_coords = [p[0] for p in points]  # First value is y
    x_coords = [p[1] for p in points]  # Second value is x

    x_min = int(min(x_coords))
    x_max = int(max(x_coords))
    y_min = int(min(y_coords))
    y_max = int(max(y_coords))

    # Crop the strip
    cropped = image[y_min:y_max, x_min:x_max]

    cv2.imwrite(f"{output_folder}\\{lead_name}.png", cropped)
    print(f"Saved {lead_name}.png - {cropped.shape[1]}x{cropped.shape[0]}")

print("Done!")