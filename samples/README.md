# Sample images

32 chest X-rays (16 normal, 16 pneumonia) copied from the **test split** of PneumoniaMNIST at 64x64,
chosen at random with a fixed seed by `python -m triage.export_samples`. `labels.csv` lists each
file's index in the test split and its dataset label.

These images are part of the held-out test set, so the model never saw them during training.

## Source and license

- Dataset: PneumoniaMNIST, part of MedMNIST v2 (https://medmnist.com/)
- License: Creative Commons Attribution 4.0 International (CC BY 4.0)
- Derived from: Kermany et al., "Identifying Medical Diagnoses and Treatable Diseases by
  Image-Based Deep Learning", Cell 2018 (pediatric chest X-rays, CC BY 4.0)
- Citation: Jiancheng Yang, Rui Shi, Donglai Wei, Zequan Liu, Lin Zhao, Bilian Ke, Hanspeter Pfister,
  Bingbing Ni. "MedMNIST v2: A large-scale lightweight benchmark for 2D and 3D biomedical image
  classification." Scientific Data, 2023.

Images were resized by MedMNIST and are not changed here.
