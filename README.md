#Ecg-digitization
This project converts scanned images of paper-based ECGs into digital time-series data. It is specifically designed to be robust against common issues like scan misalignment, page rotation, and non-standard lead layouts (e.g., 3x4, 2x6).

🎯 The Problem
Vast amounts of patient ECG data are stored on paper. To use this data for large-scale analysis, digital archiving, or machine learning, it must be converted into a digital format. Many simple conversion tools fail because they expect a perfect, non-rotated, standard-layout scan, which is rare in practice.

💡 How It Works
Our method uses a two-step pipeline to intelligently find and extract the signals.

1. Label-Guided Lead Cropping
This first step finds and isolates each of the 12 lead signals, regardless of where they are on the page.

Find Labels: We use a YOLOv8 object detection model to find the exact location of each lead's text label (e.g., 'I', 'II', 'V1') on the scanned image.

Calculate Grid: These labels are used as our reference points. By measuring the horizontal and vertical distances between them, we algorithmically calculate the precise boundaries for each lead's signal, creating a custom crop-box for every scan.

Clean and Crop: We crop out the 12 individual lead images. Each small image is then cleaned by analyzing all connected pixels and keeping only the largest group, which is the main ECG signal trace. This automatically removes grid lines and other noise.

2. Signal Digitization
This second step takes the clean, individual lead images and converts them into data points.

Signal Segmentation: Each cropped image is passed to an nnU-Net (a deep learning segmentation model). This model has been trained to "paint" or segment the exact pixels that make up the ECG signal line.

Coordinate Extraction: We then trace this segmented line, pixel by pixel, to extract a continuous sequence of (x, y) coordinates. This sequence represents the digital time-series data for that lead.

At the end of this process, a single scanned image is converted into 12 distinct digital signals, ready for analysis.

🛠️ Technology Stack
Lead Detection: YOLOv8

Signal Segmentation: nnU-Net

Core Language: Python

Image Processing: OpenCV, PIL

Deep Learning: PyTorch
