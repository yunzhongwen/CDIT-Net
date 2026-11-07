# CDIT-Net: Multi-Focus Image Fusion via Progressive Refinement, Global-Local Attention, and Cross-Domain Interaction

Official PyTorch implementation of the paper **"CDIT-Net: Multi-Focus Image Fusion via Progressive Refinement, Global-Local Attention, and Cross-Domain Interaction"** (Submitted to *The Visual Computer*).

## 1. Repository Structure
- `network2.py`: Core architecture of the proposed **CDIT-Net** (including deep residual encoder, DCP-ITB bottleneck, CDIT cross-domain soft sharing, GLA module, and progressive logit refinement).
- `Fusioning2.py`: Testing and multi-focus image fusion inference script.
- `metric/`: MATLAB objective evaluation benchmark for computing the eight fusion metrics ($Q_{NMI}$, $Q_Y$, $Q_{CB}$, $Q_{TE}$, $Q_{NCIE}$, $Q_G$, $Q_{Piella}$, and $Q_{AB/F}$).

## 2. Requirements
- Python >= 3.8, PyTorch >= 1.12.0, Torchvision, OpenCV, NumPy
- MATLAB R2022b (for running objective evaluation in `metric/`)

## 3. Datasets & Pre-trained Weights
- The public benchmark datasets used in the paper (**VOC2012**, **Lytro**, **MFFW**, **MFI-WHU**, and **Grayscale**) are available from their respective official sources cited in the manuscript.
- The pre-trained model weights and full training scripts will be released upon publication.

## 4. Contact
If you have any questions, please feel free to contact **Yunzhong Wen** (`13996513203@163.com`).
