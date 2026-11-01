# RAG techniques for image captioning (COCO)

## Dataset
- Knowledge base: 10,000 images from COCO train2017 (5 captions each)
- Validation: 100 images, Test: 100 images from COCO val2017 
- Anti-leakage: the three splits share no image (checked by ID and by image hash).

## Installation
pip install -r requirements.txt
A T4 GPU is recommended for notebook 3 (Google Colab).

## How to run (in order)
notebooks/Phase1
Results are saved to Google Drive in RAG-Phase1/.

## Reproduction
Seed = 42. Same code, same seed, same splits.

## Outputs
Plots in plots/, statistics and metadata in metadata/, index documentation in
metadata/phase1_documentation.json.

## Team
Names of the 4 members.
- Arslane Aicha Razen
- Louetri Aya
- Zerd Lyna
- Hassani Hiba
