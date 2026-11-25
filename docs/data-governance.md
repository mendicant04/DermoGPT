# Data access and reuse

DermoInstruct uses 14 sources for instruction-data construction. Four additional sources are used only for held-out DermoBench evaluation. The tables below summarize Table IV of the TMI revision, with provider links and clarifications where available.

[DermoInstruct](https://huggingface.co/datasets/mendicant04/DermoInstruct) provides the 646,018-entry supervised fine-tuning snapshot. [DermoBench](https://huggingface.co/datasets/mendicant04/DermoBench) provides the evaluation resources. The [DermoGPT-RL checkpoint](https://huggingface.co/mendicant04/DermoGPT-RL) is intended for research and benchmark evaluation. The Hugging Face pages list CC BY-NC 4.0 and their respective access procedures.

Provider terms govern the original images and any source-specific reuse. Obtain original images directly from their providers and follow the terms that apply to the files and intended use. Project release terms apply only to materials the authors are authorized to license. The project treats clinical deployment, autonomous diagnosis, and direct patient-facing use as out of scope.

**Key.** IC means informed consent. NR means the reviewed source documentation did not report the item. UV means the item has not been verified. Ethics and consent entries describe the source studies.

## DermoInstruct construction sources

| Source | Access or source documentation | License or reuse terms | Source ethics / consent |
| --- | --- | --- | --- |
| Daffodil | [Mendeley Data, version 1](https://data.mendeley.com/datasets/3hckgznc67/1) | CC BY 4.0 | NR / NR |
| DermNet | [Kaggle Dermnet listing](https://www.kaggle.com/datasets/shubhamgoel27/dermnet) | The listing reports CC BY-NC-ND 4.0. Project-specific rights and the origin of the exact image files require separate verification. See the source note below. | NR / NR |
| Fitzpatrick17k | [Original project and access instructions](https://github.com/mattgroh/fitzpatrick17k) | CC BY-NC-SA 3.0 as recorded in Table IV. Consult the project for current access and the terms covering underlying images. | NR / NR |
| ISIC Archive | [ISIC Archive](https://gallery.isic-archive.com/) | Image-specific Creative Commons terms. Check the license on each relevant archive record. | Contributor terms |
| MIDAS | [Stanford AIMI access page](https://aimi.stanford.edu/datasets/mra-midas-Multimodal-Image-Dataset-for-AI-based-Skin-Cancer) | Stanford AIMI access terms, noncommercial | IRB / written IC |
| PAD-UFES-20 | [Mendeley Data, version 1](https://data.mendeley.com/datasets/zr7vgbcyr2/1) | CC BY 4.0 | IRB / IC |
| PASSION | [Official project and download](https://passionderm.github.io/) | Custom, noncommercial. Follow the provider's access and reuse terms. | Site ethics / IC |
| PUMCH-ISD | [Source study](https://doi.org/10.1016/j.jid.2025.03.034) | NR. Consult the source authors for access and reuse. | IRB / written IC |
| SCIN | [Official dataset](https://github.com/google-research-datasets/scin) and [license](https://github.com/google-research-datasets/scin/blob/main/LICENSE) | SCIN dataset license | IRB / online IC |
| SD-198 | [Source benchmark description](https://xiaoxiaosun.com/docs/2016-eccv-sd198.pdf) | NR | NR / NR |
| BCN20000 | [ISIC Challenge dataset index](https://challenge.isic-archive.com/data/) | CC BY 4.0 as recorded in Table IV. Check the archive release and image records for applicable terms. | IRB / IC |
| HAM10000 | [Harvard Dataverse](https://doi.org/10.7910/DVN/DBW86T) | CC BY-NC 4.0 | IRB / NR |
| Derm12345 | [Harvard Dataverse](https://doi.org/10.7910/DVN/DAXZ7P) | CC BY 4.0 | IRB / waived IC |
| MILK10k | [ISIC Archive dataset record](https://doi.org/10.34970/648456) | CC BY-NC 4.0 | IRB / IC not sought |

## Held-out evaluation sources

| Source | Access or source documentation | License or reuse terms | Source ethics / consent |
| --- | --- | --- | --- |
| Derm1M | [Official dataset and access instructions](https://huggingface.co/datasets/redlessone/Derm1M) | CC BY-NC 4.0. Follow the conditions for the relevant subset. | NR / NR |
| DDI | [Stanford AIMI access page](https://aimi.stanford.edu/datasets/ddi-diverse-dermatology-images) | Research agreement | IRB / NR |
| Derm7pt | [SFU data download](https://derm.cs.sfu.ca/Download.html) | CC BY-NC-ND 4.0 for images and metadata. Access requires the provider's request form. | NR / NR |
| SNU | [Figshare, version 12](https://doi.org/10.6084/m9.figshare.6454973.v12) | CC BY 4.0 according to the [version 12 metadata](https://api.figshare.com/v2/articles/6454973/versions/12) | UV / UV |

## DermNet provenance

The Kaggle mirror and the terms supplied by an original image provider are separate records. [DermNet New Zealand's image terms](https://dermnetnz.org/image-licence) apply to images obtained from its site and describe a separate licensing route for AI use. The public records reviewed here do not verify the exact provider provenance or any project-specific agreement for the files used by DermoInstruct.
