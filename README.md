# BPRAA AI
## AI-Assisted Bloodstain Pattern Analysis with Generalization Across Unseen Conditions

BPRAA AI is a research-oriented forensic computer vision project for analysing bloodstain patterns using traditional machine-learning and deep-learning models.

The central aim is not only to obtain high in-domain classification accuracy, but also to evaluate whether models remain reliable when forensic imaging conditions change.

The project evaluates robustness under unseen conditions such as:

- Different surfaces and substrates
- Indoor and outdoor environments
- Lighting changes
- Camera/viewing-angle changes
- Perspective changes
- Image degradation, blur, noise, compression, and low resolution
- Differences between independently collected datasets

---

## 🔬 Research Focus

### Planned Model Families

- Traditional machine learning: Random Forest and XGBoost
- CNN models: ResNet and MobileNet
- Transformer models: Vision Transformer (ViT)
- Supporting methods: segmentation, feature extraction, transfer learning, confidence estimation, and explainable AI

---

# 📊 Dataset Sources

> **Important:** The datasets are not redistributed in this repository. Download them from their official source pages and follow their licences, terms of use, and citation requirements.

## 1. Passive Bloodstain / Drip Pattern Dataset

**Dataset paper:**  
Kannan, S., Machong, S. M., Lewis, P. R., & Stotesbury, T. (2025). *A dataset of drip patterns for teaching and research purposes in forensic bloodstain pattern analysis*. Data in Brief, 59, 111352.

- **DOI:** https://doi.org/10.1016/j.dib.2025.111352
- **Paper:** https://pmc.ncbi.nlm.nih.gov/articles/PMC11847719/
- **Dataset:** https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.5683/SP3/YSTDGI
- **Metadata repository:** https://gitlab.com/f4301/passive-blood-stains-metadata

**Use in BPRAA AI:** Primary dataset for evaluating cross-surface generalization.

---

## 2. Impact Beating Spatter Dataset

**Dataset paper:**  
Attinger, D., et al. (2018). *A data set of bloodstain patterns for teaching and research in bloodstain pattern analysis: Impact beating spatters*. Data in Brief.

- **NIJ source:** https://nij.ojp.gov/library/publications/data-set-bloodstain-patterns-teaching-and-research-bloodstain-pattern-1
- **Paper / Publisher:** https://www.sciencedirect.com/science/article/pii/S2352340918306680
- **DOI:** https://doi.org/10.1016/j.dib.2018.02.070

**Use in BPRAA AI:** External validation and cross-dataset testing.

---

## 3. Gunshot Backspatter Dataset

**Dataset paper:**  
Attinger, D., et al. (2019). *A data set of bloodstain patterns for teaching and research in bloodstain pattern analysis: Gunshot backspatters*. Data in Brief.

- **NIJ source:** https://nij.ojp.gov/library/publications/data-set-bloodstain-patterns-teaching-and-research-bloodstain-pattern-2
- **Paper / Publisher:** https://www.sciencedirect.com/science/article/pii/S2352340918314689
- **DOI:** https://doi.org/10.1016/j.dib.2018.03.007

**Use in BPRAA AI:** External validation and pattern-type transfer evaluation.

---

## 4. MFRC Bloodstain Pattern Analysis Video Collection

- **Archive:** https://alvideo.ameslab.gov/archive/bpa-videos/
- **NIST bibliography:** https://www.nist.gov/system/files/documents/forensics/Annotated-Bibliography-Bloodstain.pdf
- **Related code:** https://github.com/Zdong104/Bloodstain-Analysis-AI-Tool
- **Related paper:** https://arxiv.org/abs/2308.13979

**Use in BPRAA AI:** Video-frame extraction, segmentation, droplet formation, impact, and angle-related experiments.

---

## 5. Iowa State Bloodstain Pattern and Fluid Dynamics Dataset Collection

- **Dataset collection:** https://iastate.figshare.com/collections/Blood_stain_pattern_and_fluid_dynamics_datasets/5672977
- **Iowa State repository category:** https://iastate.figshare.com/categories/Engineering/26212

**Use in BPRAA AI:** Feature extraction, contour analysis, morphology estimation, and external validation.

---

## 6. BloodNet Benchmark Dataset

- **Dataset:** https://figshare.com/articles/dataset/BloodNet_An_attention-based_deep_network_for_accurate_efficient_and_costless_bloodstain_time_since_deposition_inference/21291825
- **Related paper:** https://academic.oup.com/bib/article/24/1/bbac557/6960974

**Use in BPRAA AI:** Transfer learning and robustness to bloodstain ageing.

> This dataset primarily focuses on time-since-deposition inference and is not a direct replacement for BPA pattern-classification datasets.

---

## 7. Hyperspectral Bloodstain Dataset

- **Dataset:** https://zenodo.org/record/3984905
- **Related implementation:** https://github.com/MHassaanButt/FCHCNN-for-HSIC
- **Related paper:** https://pmc.ncbi.nlm.nih.gov/articles/PMC7700311/

**Use in BPRAA AI:** Advanced blood-versus-lookalike classification and robustness experiments using hyperspectral data.

---

## 8. Roboflow Bloodstain Segmentation Dataset

- **Segmentation dataset:** https://universe.roboflow.com/bloodstain-segmentation/bloodstain-segmentation
- **Alternative detection dataset:** https://universe.roboflow.com/ghalya-hxgho/bloodstain-z2fox

**Use in BPRAA AI:** Bloodstain detection, segmentation, and stain-region extraction.

> Verify dataset licences, labels, and usage permissions before using the data in final experiments.

---

## 9. Kaggle Bloodstain Pattern Analysis Dataset

- **Dataset:** https://www.kaggle.com/datasets/zhasan557/bloodstain-pattern-analysis
- **Alternative BPA scans:** https://www.kaggle.com/datasets/marcoferrarini/bpa-scans

**Use in BPRAA AI:** Early prototyping, preprocessing, data-loader testing, and preliminary CNN experiments.

> Before using Kaggle data for final reported results, verify the original image sources, licences, duplicates, labels, and metadata.

---

# 📚 10 Research Papers Collected

## 1. Explainable Deep Learning in Bloodstain Pattern Analysis

**Title:** *Explainable Deep Learning in Bloodstain Pattern Analysis: A Pilot Study Using Convolutional Neural Networks with Saliency Maps*

**Authors:** V. P. Chantzi, J. Millington, E. Mariconti, et al.  
**Year:** 2026  
**Journal:** Forensic Science International: Digital Investigation

- **Publisher:** https://www.sciencedirect.com/science/article/pii/S2589871X26000550

**Relevance:** Explainable deep learning, CNN classification, saliency maps, and forensic decision support.

---

## 2. Classification of Wipe and Swipe Bloodstain Patterns

**Title:** *A Preliminary Investigation into the Classification of Wipe and Swipe Bloodstain Patterns Between Human and Artificial Intelligence*

**Authors:** G. Griffiths and D. J. Parker  
**Year:** 2026  
**Journal:** Journal of Forensic Sciences

- **DOI:** https://doi.org/10.1111/1556-4029.70225
- **Publisher:** https://onlinelibrary.wiley.com/doi/10.1111/1556-4029.70225

**Relevance:** AI classification of wipe and swipe patterns and multi-class BPA classification.

---

## 3. Machine Learning for Blood Pattern Classification

**Title:** *From Images to Detection: Machine Learning for Blood Pattern Classification*

**Authors:** Yilin Li and Weining Shen  
**Year:** 2025  
**Journal:** Forensic Science International

- **DOI:** https://doi.org/10.1016/j.forsciint.2025.112558
- **Preprint:** https://arxiv.org/abs/2501.02151

**Relevance:** Random Forest, XGBoost, blood-pattern classification, and handcrafted morphological features.

---

## 4. Drip Pattern Dataset

**Title:** *A Dataset of Drip Patterns for Teaching and Research Purposes in Forensic Bloodstain Pattern Analysis*

**Authors:** S. Kannan, S. M. Machong, P. R. Lewis, and T. Stotesbury  
**Year:** 2025  
**Journal:** Data in Brief

- **DOI:** https://doi.org/10.1016/j.dib.2025.111352
- **Paper:** https://pmc.ncbi.nlm.nih.gov/articles/PMC11847719/

**Relevance:** Multi-surface drip-pattern dataset and condition-based/cross-surface testing.

---

## 5. Passive Bloodstain Morphology Across Surface Textures

**Title:** *Analysis of Passive Bloodstain Morphology Across Surface Textures and Drop Heights Using Deep Learning*

**Authors:** S. Jomon, B. T. Bastian, and M. M. Joseph  
**Year:** 2026  
**Journal:** Forensic Science International

- **DOI:** https://doi.org/10.1016/j.forsciint.2026.112890
- **Paper:** https://pubmed.ncbi.nlm.nih.gov/41707408/

**Relevance:** Surface texture, drop height, morphology changes, and generalization across substrates.

---

## 6. Impact of Image Size on BPA

**Title:** *Technical Note: The Impact of Image Size on Bloodstain Pattern Analysis Using Machine Learning*

**Authors:** Ainaz Alavi, Theresa Stotesbury, and Peter R. Lewis  
**Year:** 2025  
**Journal:** Forensic Science International

- **DOI:** https://doi.org/10.1016/j.forsciint.2025.112728
- **Paper:** https://pubmed.ncbi.nlm.nih.gov/41242137/

**Relevance:** Image size, resolution, image degradation, and ML performance.

---

## 7. One-Class Classification for Bloodstain Detection

**Title:** *A Comparative Study of One-Class Classification Methods for Bloodstain Detection in Hyperspectral Forensic Imaging*

**Year:** 2026  
**Journal:** Neural Computing and Applications

- **DOI:** https://doi.org/10.1007/s00521-025-11825-y
- **Paper:** https://link.springer.com/article/10.1007/s00521-025-11825-y

**Relevance:** Detection under unfamiliar conditions, anomaly detection, and uncertainty handling.

---

## 8. Transformer-Based Blood Pattern Classification

**Title:** *Blood Pattern Classification Using Transformer-Based Image Features*

**Year:** 2026  
**Publisher:** ACM Digital Library

- **DOI:** https://doi.org/10.1145/3812734.3813710
- **Paper:** https://dl.acm.org/doi/10.1145/3812734.3813710

**Relevance:** Vision Transformer representations and CNN-vs-transformer comparison.

---

## 9. Deep Learning for Bloodstain Patterns and Footprint Impressions

**Title:** *Enhancing Detection of Bloodstain Patterns and Footprint Impressions Using Deep Learning*

**Year:** 2026  
**Publisher:** IEEE

- **DOI:** https://doi.org/10.1109/ISDFS69419.2026.11459089
- **Paper:** https://ieeexplore.ieee.org/document/11459089

**Relevance:** Deep learning, Vision Transformer methods, and automated forensic evidence detection.

---

## 10. Automatic Classification of Bloodstains

**Title:** *Automatic Classification of Bloodstains with Deep Learning Methods*

**Year:** 2022  
**Journal:** KI – Künstliche Intelligenz

- **DOI:** https://doi.org/10.1007/s13218-022-00760-y
- **Paper:** https://link.springer.com/article/10.1007/s13218-022-00760-y

**Relevance:** Foundational CNN-based bloodstain classification and baseline model development.

---

# 🧪 Dataset Selection for the Core Project

| Project Stage | Dataset | Purpose |
|---|---|---|
| Main training & unseen-surface testing | Passive Bloodstain / Drip Pattern Dataset | Surface and environment generalization |
| Segmentation / stain extraction | Roboflow or MFRC frames | Bloodstain region extraction |
| External validation | Impact Beating Spatter Dataset | Cross-dataset performance |
| External validation | Gunshot Backspatter Dataset | Pattern-type transfer |
| Advanced extension | MFRC High-Speed Videos | Perspective, impact, and frame analysis |
| Advanced extension | BloodNet | Transfer learning / ageing robustness |
| Advanced extension | Hyperspectral Dataset | Blood-versus-lookalike detection |

---

# 📈 Evaluation Protocol

### In-Domain Testing

The model is trained and evaluated on images from similar conditions.

Example:

- Training: paper and ceramic-tile images
- Testing: paper and ceramic-tile images

### Unseen-Condition Testing

The model is trained on selected conditions and evaluated on conditions not used during training.

Example:

- Training: paper and ceramic tile
- Testing: grass and snow

### Image-Quality Testing

- Training: original-quality images
- Testing: low-resolution, blurred, noisy, and JPEG-compressed images

### Generalization Gap

**Generalization Gap = In-Domain F1-score − Unseen-Condition F1-score**

A smaller generalization gap indicates better robustness.

---

# 📊 Metrics

The project will report:

- Accuracy
- Precision
- Recall
- Macro F1-score
- Weighted F1-score
- Balanced accuracy
- Confusion matrix
- Per-class recall
- Generalization gap
- Calibration/confidence metrics
- IoU and Dice score, if segmentation is implemented

---

# ⚖️ Ethical and Research Use Notice

This repository is intended for academic and forensic decision-support research only.

The proposed system must not be treated as a standalone forensic conclusion or replacement for qualified bloodstain-pattern analysts. Model outputs should be interpreted as decision-support information and should include uncertainty estimates, data limitations, and explainability visualizations where possible.

All datasets, images, annotations, and papers remain subject to the original source's licence, terms of use, ethics requirements, and citation rules.
