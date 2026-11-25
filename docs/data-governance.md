# Data access and reuse

DermoInstruct uses 14 training sources. Derm1M, DDI, Derm7pt, and SNU are used solely for held-out DermoBench evaluation and are excluded from DermoInstruct training.

[DermoInstruct annotations](https://huggingface.co/datasets/mendicant04/DermoInstruct), [DermoBench evaluation resources](https://huggingface.co/datasets/mendicant04/DermoBench), and the [DermoGPT-RL checkpoint](https://huggingface.co/mendicant04/DermoGPT-RL) are available for research use under CC BY-NC 4.0. The checkpoint is intended for research and benchmark evaluation. The code is released under [Apache-2.0](../LICENSE). Original source images remain governed by their providers' licenses and access conditions.

The tables reproduce the source licenses, reuse conditions, and ethics and consent entries in Tables 3 and 12 of our IEEE TMI response to reviewers.

**Key.** IC means informed consent. NR means not reported in reviewed documentation. UV means not verified. CC terms retain attribution (BY), noncommercial use (NC), share-alike (SA), or no sharing of adaptations (ND), as applicable.

## DermoInstruct training sources

| Source | Access or source documentation | License / reuse | Ethics / consent |
| --- | --- | --- | --- |
| Daffodil | [Mendeley Data, version 1](https://data.mendeley.com/datasets/3hckgznc67/1) | CC BY 4.0 | NR / NR |
| DermNet | [Kaggle Dermnet listing](https://www.kaggle.com/datasets/shubhamgoel27/dermnet) | CC BY-NC-ND 4.0 | NR / NR |
| Fitzpatrick17k | [Original project and access instructions](https://github.com/mattgroh/fitzpatrick17k) | CC BY-NC-SA 3.0 | NR / NR |
| ISIC Archive | [ISIC Archive](https://gallery.isic-archive.com/) | Image-specific CC | Contributor terms |
| MIDAS | [Stanford AIMI access page](https://aimi.stanford.edu/datasets/mra-midas-Multimodal-Image-Dataset-for-AI-based-Skin-Cancer) | AIMI terms, NC | IRB / written IC |
| PAD-UFES-20 | [Mendeley Data, version 1](https://data.mendeley.com/datasets/zr7vgbcyr2/1) | CC BY 4.0 | IRB / IC |
| PASSION | [Official project and download](https://passionderm.github.io/) | Custom, NC | Site ethics / IC |
| PUMCH | [Source study](https://doi.org/10.1016/j.jid.2025.03.034) | NR | IRB / written IC |
| SCIN | [Official dataset](https://github.com/google-research-datasets/scin) and [license](https://github.com/google-research-datasets/scin/blob/main/LICENSE) | SCIN license | IRB / online IC |
| SD-198 | [Source benchmark description](https://xiaoxiaosun.com/docs/2016-eccv-sd198.pdf) | NR | NR / NR |
| BCN20000 | [ISIC Challenge dataset index](https://challenge.isic-archive.com/data/) | CC BY 4.0 | IRB / IC |
| HAM10000 | [Harvard Dataverse](https://doi.org/10.7910/DVN/DBW86T) | CC BY-NC 4.0 | IRB / NR |
| Derm12345 | [Harvard Dataverse](https://doi.org/10.7910/DVN/DAXZ7P) | CC BY 4.0 | IRB / waived IC |
| MILK10k | [ISIC Archive dataset record](https://doi.org/10.34970/648456) | CC BY-NC 4.0 | IRB / IC not sought |

## Held-out evaluation sources

| Source | Access or source documentation | License / reuse | Ethics / consent |
| --- | --- | --- | --- |
| Derm1M | [Official dataset and access instructions](https://huggingface.co/datasets/redlessone/Derm1M) | CC BY-NC 4.0 | NR / NR |
| DDI | [Stanford AIMI access page](https://aimi.stanford.edu/datasets/ddi-diverse-dermatology-images) | Research agreement | IRB / NR |
| Derm7pt | [SFU data download](https://derm.cs.sfu.ca/Download.html) | CC BY-NC-ND 4.0 | NR / NR |
| SNU | [Figshare, version 12](https://doi.org/10.6084/m9.figshare.6454973.v12) | CC BY 4.0 | UV / UV |
