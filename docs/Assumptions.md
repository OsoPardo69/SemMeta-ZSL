**Comparison of Zero-shot methods and their key qualifications.**

| Method | Prototype Source | Unseen Signal | Data Preprocessing | Calibration | Tune Parameters | Key qualification |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **SMETA-ZSL /ou** | Yes<br>CTI Embeddings | No | Yes | Yes | 6 | Only method using *neither* unlabeled nor 1-shot unseen data |
| MZSL | Yes<br>Tabular Data | Unlabeled | Yes | Yes | 4 | Unseen attributes computed from unseen *test* samples (implicit transduction) |
| FL-ZSL | Yes<br>CTI Embeddings | No | Yes | Yes | 2 | 3-client federated split |
| TZSL | Yes<br>Tabular Data | Unlabeled | Yes | Yes | 0 | Explicit: unlabeled $X_{test}$ in VQ-VAE training |
| CVAE-ZSL | Yes<br>Tabular Data | Unlabeled | Yes | Yes | 3 | Unseen attrs from test (implicit transduction) |
| CLIP-Decoder | Yes<br>CTI Embeddings | Unlabeled | Yes | Yes | 0 | Trained on massive unlabeled data |
| SMELL | Yes<br>CTI Embeddings | No | No | Yes | 0 | Objective is to categorize malware/benign |
| P2T | Yes<br>CTI Embeddings | Unlabeled | Yes | No | 0 | LLM prompting for tabular features. Transductive In Context Learning |
| ZET-LLM | Yes<br>Class Names | No | Yes | Yes | 1 | Acts as feature extractor. Prompts an LLM with the class names and the tabular data |
| Proto LLM | Yes<br>Class Names | No | No | Yes | 1 | Acts as prototype generator. Prompts an LLM with the class name and a sample description |
