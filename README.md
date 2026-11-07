# CDIT-Net: Multi-Focus Image Fusion via Progressive Refinement, Global-Local Attention, and Cross-Domain Interaction

Official PyTorch implementation of the paper **"CDIT-Net: Multi-Focus Image Fusion via Progressive Refinement, Global-Local Attention, and Cross-Domain Interaction"**.

## 1. Repository Structure
- `network2.py`: Core architecture of the proposed **CDIT-Net**.
- `Fusioning2.py`: Testing and multi-focus image fusion inference script.
- `metric/`: MATLAB objective evaluation package for computing the eight fusion metrics ($Q_{NMI}$, $Q_Y$, $Q_{CB}$, $Q_{TE}$, $Q_{NCIE}$, $Q_G$, $Q_{Piella}$, and $Q_{AB/F}$).

## 2. Requirements
- Python >= 3.8, PyTorch >= 1.12.0, Torchvision, OpenCV, NumPy
- MATLAB R2022b (for objective metric evaluation in `metric/`)

## 3. Contact
If you have any questions, please feel free to contact **Yunzhong Wen** (`13996513203@163.com`).
