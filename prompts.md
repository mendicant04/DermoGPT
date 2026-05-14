# DermoGPT Prompts

This document consolidates the prompt templates used in DermoGPT, DermoInstruct, and DermoBench.
It is intended as a standalone prompt reference for readers who cannot access a paper appendix.

Sources:

- Benchmark user prompts: `dataset_final/benchmark/**`
- Inference-time format prompts: `src/infer/**`
- Training-time task system prompts: `src/constants.py`
- LLM-as-a-Judge prompts: `dataset_final/benchmark/task*/**/*judge.py`
- Paper appendix reference: Appendix C and D of `DermoGPT.pdf`

## 1. Notation

- `<image>` marks the image token in dataset JSON/JSONL files.
- `{options}` or `{options_list}` is filled with shuffled answer choices.
- `{task_prompt}`, `{reference}`, and `{candidate}` are filled by evaluation scripts.
- Clinical images use the SkinCon morphology schema.
- Dermoscopic images use the Derm7pt morphology schema.

## 2. Closed Morphology Vocabularies

### 2.1 SkinCon

```text
Abscess
Acuminate
Atrophy
Black
Blue
Brown(Hyperpigmentation)
Bulla
Burrow
Comedo
Crust
Cyst
Dome-shaped
Erosion
Erythema
Excoriation
Exophytic/Fungating
Exudate
Fissure
Flat topped
Friable
Gray
Induration
Lichenification
Macule
Nodule
Papule
Patch
Pedunculated
Pigmented
Plaque
Poikiloderma
Purple
Purpura/Petechiae
Pustule
Salmon
Scale
Scar
Sclerosis
Telangiectasia
Translucent
Ulcer
Umbilicated
Vesicle
Warty/Papillomatous
Wheal
White(Hypopigmentation)
Xerosis
Yellow
```

### 2.2 Derm7pt

```text
pigment_network: absent | typical | atypical
blue_whitish_veil: absent | present
vascular_structures: absent | arborizing | comma | hairpin | within regression | wreath | dotted | linear irregular
pigmentation: absent | diffuse regular | localized regular | diffuse irregular | localized irregular
streaks: absent | regular | irregular
dots_and_globules: absent | regular | irregular
regression_structures: absent | blue areas | white areas | combinations
```

## 3. DermoBench Task Prompts

### 3.1 Task 1.1: Open-Ended Morphological Description

The benchmark samples one of the following user prompts.

```text
<image>
Provide a comprehensive and detailed clinical morphological description of the skin lesion in the image.
```

```text
<image>
Write a detailed morphological examination report for this image.
```

```text
<image>
Summarize all important morphological features visible in this image.
```

```text
<image>
Briefly describe the main lesion morphology and its most prominent features shown in the image.
```

```text
<image>
Describe the lesion based on the 7-point checklist (pigment network, blue-whitish veil, vascular structures, pigmentation, streaks, dots/globules, and regression structures).
```

```text
<image>
Describe in detail the overall pattern and key structures seen in this dermoscopic image.
```

### 3.2 Task 1.2: Morph-Grounded Description

Task 1.2 uses the Task 1.1 description prompts, then appends a schema-specific output instruction.

Clinical/SkinCon suffix:

```text
Before any reasoning, first output a <morph> JSON using the SkinCon schema, then provide exactly ONE detailed paragraph.
```

Dermoscopic/Derm7pt suffix:

```text
Before any reasoning, first output a <morph> JSON using the Derm7pt schema, then provide exactly ONE detailed paragraph.
```

Example composed SkinCon prompt:

```text
<image>
Summarize all important morphological features visible in this image.

Before any reasoning, first output a <morph> JSON using the SkinCon schema, then provide exactly ONE detailed paragraph.
```

Example composed Derm7pt prompt:

```text
<image>
Describe the lesion based on the 7-point checklist (pigment network, blue-whitish veil, vascular structures, pigmentation, streaks, dots/globules, and regression structures).

Before any reasoning, first output a <morph> JSON using the Derm7pt schema, then provide exactly ONE detailed paragraph.
```

### 3.3 Task 1.3: Derm7pt Attribute MCQA

Each question appends answer options from the valid labels for the queried Derm7pt attribute.

Prompt stems:

```text
Does the lesion exhibit a pigment network? If so, how would you characterize it?
Describe the nature of streaks in the dermoscopic view.
Evaluate the pigmentation within the lesion and select the most accurate description.
Does the dermoscopic image show any signs of regression, such as specific color changes?
How are the dots and globules characterized in this skin lesion?
Does the lesion show evidence of a blue whitish veil?
From the list below, select the term that best describes the vascularity of the lesion.
Are streaks present in the image? If so, are they regular or irregular?
How would you describe the specific pigmentation pattern in the lesion?
Are regression structures visible, and if so, of what type?
Which of the following specific vascular structures are observed in the lesion?
Based on the dermoscopy, classify the pigment network.
What is the characteristic of the streaks, if any, in this lesion?
Classify the detailed pigmentation of the skin lesion shown in the image.
Identify the specific type of regression structures present in the lesion.
Identify the precise vascular pattern present in the dermoscopic image.
What is the status of the pigment network in this dermoscopic image?
Classify the pattern of dots and globules shown in the image.
Analyze the dots and globules within the lesion.
Is a blue whitish veil visible in the dermoscopic examination?
Evaluate the presence of a blue whitish veil in the provided image.
```

Example:

```text
<image>
Does the lesion exhibit a pigment network? If so, how would you characterize it?
A) Absent
B) Typical
C) Atypical
```

### 3.4 Task 1.4: SkinCon Attribute MCQA

The benchmark samples one of the following stems and fills four SkinCon options.

```text
Considering the characteristics of the skin lesion shown, which of the following descriptions applies?
Please identify the correct dermatological term that describes a feature in this image from the choices below.
Which of the following clinical features is present in the image?
Based on the visual evidence provided, which of the following findings can be observed on the lesion?
```

Example:

```text
<image>
Which of the following clinical features is present in the image?
A) Plaque
B) Abscess
C) Friable
D) Black
```

### 3.5 Task 2.1 and Task 4: Diagnosis MCQA

Task 2.1 uses ID diagnosis questions. Task 4 uses the same MCQA style on the DDI fairness subset.

Prompt stems:

```text
Considering the clinical presentation of the skin lesion in the image, which of the following is the most likely diagnosis?
Based on the skin lesion shown in this image, please select the most accurate diagnosis from the options below.
Observe this skin image. Which of the following diagnoses is the most likely?
Which of the following diagnoses best matches the skin condition shown in this image?
```

Example 4-choice prompt:

```text
<image>
Based on the skin lesion shown in this image, please select the most accurate diagnosis from the options below.
A) Nevus / Mole / Melanocytic Nevus
B) Dilated Pore Of Winer
C) Trichofolliculoma
D) Cyst
```

Example 25-choice prompt:

```text
<image>
Considering the clinical presentation of the skin lesion in the image, which of the following is the most likely diagnosis?
A) Milia
B) Trichofolliculoma
C) Supernumerary Nipple
...
Y) Cyst
```

Inference scripts append the following control sentence to MCQA prompts:

```text
NOTE: Respond with ONLY the letter of your choice.
```

### 3.6 Task 2.2: Hierarchical Diagnosis

System prompt:

```text
You are an expert dermatology visual question answering assistant.

You will be shown a dermoscopic or clinical skin lesion image and asked one or more
multiple-choice classification questions in a hierarchical way about the SAME image.

For every question you receive, follow these rules VERY STRICTLY:

1. The user message will describe a question and then give a list of answer options,
   usually in the following form:
   - "Option 1", "Option 2", "Option 3", ...
   - or similar comma-separated options in double quotes.

2. Your task is to choose the single BEST option from this list.

3. Your ENTIRE reply must be EXACTLY one of the option strings, copied verbatim:
   - Do NOT add any other words.
   - Do NOT add quotes around your answer.
   - Do NOT prefix with letters such as "A)" or "B)".
   - Do NOT explain your reasoning.
   - Do NOT output multiple options.
   - Do NOT output anything like "The correct answer is ...".
   The output should be ONLY the chosen option text, nothing else.

4. If you are unsure, you must still pick the single most plausible option and
   output ONLY that option string.

5. You may receive corrective messages like:
   "Actually, that's incorrect. The correct category should be 'X'. Let's proceed with that."
   These messages tell you what the correct option is for the PREVIOUS question.
   Use this information as context, but still obey rules (1)-(4) for the NEXT question.

6. You will be asked several questions about the same image in sequence.
   Answer each question independently following the format rules above.
```

Top-level prompt templates:

```text
Based on the clinical image, identify the most fitting major dermatological category from the following list: {options_list}.
Observe the skin lesion. Which of these high-level classifications best describes it? Here are the possibilities: {options_list}.
Please provide a broad categorization for the skin condition shown. Your answer should be one of the following: {options_list}.
```

Sub-level prompt templates:

```text
Correct, the condition is a form of '{parent_category}'. Now, specify the sub-category from this list: {options_list}.
Proceeding from '{parent_category}', which of the following groups does this lesion belong to? {options_list}.
Understood. Let's refine the diagnosis within '{parent_category}'. Please choose the most accurate description from the following: {options_list}.
```

Final-level prompt templates:

```text
We've classified this under '{parent_category}'. Now, provide the definitive diagnosis from the choices available: {options_list}.
Excellent. To finalize, please state the specific diagnosis for '{parent_category}', which should be one of the following: {options_list}.
Perfect. Based on our hierarchical classification ending with '{parent_category}', please identify the definitive diagnosis from this list: {options_list}.
```

Correction prompt templates:

```text
Actually, that's incorrect. A closer look reveals features more consistent with '{correct_choice}'. Please correct the path.
That's not quite right. The correct category here should be '{correct_choice}'. Let's proceed with that.
Incorrect. The diagnosis should be '{correct_choice}'. Continue from this category.
```

### 3.7 Task 3.1: CoT Reasoning

The benchmark samples one of the following user prompts.

```text
<image>
What skin condition is depicted in this image? Justify your diagnosis with a clear rationale.
```

```text
<image>
Based on the provided image, what is the most likely diagnosis? Please provide a detailed reasoning process before giving the final answer.
```

```text
<image>
Analyze the clinical presentation in this image. What is your differential diagnosis, and what is the final conclusion? Explain your reasoning.
```

```text
<image>
Examine the image carefully. Please provide a step-by-step reasoning process to arrive at a dermatological diagnosis.
```

Inference-time system prompt:

```text
You are a dermatology VQA assistant.
Output EXACTLY TWO blocks in this order and nothing else:
1) <reasoning> Provide a concise, step-by-step, image-grounded chain-of-thought reasoning process. No probabilities, disclaimers, or instructions.</reasoning>
2) <final_diagnosis>ONE most likely diagnosis (free-text clinical term). No extra words.</final_diagnosis>

Strict rules:
- Do NOT echo the question; do NOT add markdown, code fences, or labels such as 'Answer:'.
- If uncertain, still pick the single most likely diagnosis based on visible cues.
- Do NOT include patient management, tests, or treatments.
```

### 3.8 Task 3.2: Morph-Grounded Reasoning

Task 3.2 uses the Task 3.1 reasoning prompts, then appends a schema-specific output instruction.

Clinical/SkinCon suffix:

```text
Then output EXACTLY three blocks in this order and nothing else:
<reasoning>your step-by-step, image-grounded reasoning</reasoning>
<morph>{STRICT JSON using the SkinCon schema; list ONLY present features; valid JSON}</morph>
<final_diagnosis>ONE label from our taxonomy</final_diagnosis>
```

Dermoscopic/Derm7pt suffix:

```text
Then output EXACTLY three blocks in this order and nothing else:
<reasoning>your step-by-step, image-grounded reasoning</reasoning>
<morph>{STRICT JSON using the Derm7pt schema; exactly one value per field; valid JSON}</morph>
<final_diagnosis>ONE label from our taxonomy</final_diagnosis>
```

Inference-time SkinCon system prompt:

```text
You are a dermatology VQA assistant.
Output EXACTLY THREE blocks in this order and nothing else:
1) <reasoning> Provide a concise, step-by-step, image-grounded chain-of-thought reasoning process.</reasoning>
2) <morph>
{
  "morphological_features_skincon": [ ... ]
}
</morph>
3) <final_diagnosis>ONE label from the provided taxonomy</final_diagnosis>

Strict rules:
- In <morph>, return ONLY a valid JSON object with EXACTLY one key "morphological_features_skincon".
- The value is an array (possibly empty) of strings chosen ONLY from the SkinCon closed set.
- Include ONLY features visibly present; if none, use []. Sort the array alphabetically.
- Do NOT add code fences or extra text anywhere.
```

Inference-time Derm7pt system prompt:

```text
You are a dermoscopy VQA assistant.
Output EXACTLY THREE blocks in this order and nothing else:
1) <reasoning> Provide a concise, step-by-step, image-grounded chain-of-thought reasoning process.</reasoning>
2) <morph>
{
  "morphological_features_Derm7pt": {
    "pigment_network": "absent" | "typical" | "atypical",
    "blue_whitish_veil": "absent" | "present",
    "vascular_structures": "absent" | "arborizing" | "comma" | "hairpin" | "within regression" | "wreath" | "dotted" | "linear irregular",
    "pigmentation": "absent" | "diffuse regular" | "localized regular" | "diffuse irregular" | "localized irregular",
    "streaks": "absent" | "regular" | "irregular",
    "dots_and_globules": "absent" | "regular" | "irregular",
    "regression_structures": "absent" | "blue areas" | "white areas" | "combinations"
  }
}
</morph>
3) <final_diagnosis>ONE label from the provided taxonomy</final_diagnosis>

Strict rules:
- Use EXACT lowercase_snake_case keys shown above; output ALL keys.
- If a structure is not present, set it to "absent".
- Do NOT add code fences or extra text anywhere.
```

## 4. DermoInstruct Data Synthesis Prompts

### 4.1 Clinical Morphology JSON Prompt: SkinCon

System prompt:

```text
You are an expert in dermatology. Your task is to perform a detailed visual analysis of a provided skin lesion image (clinical or dermoscopic). You will be given an image of a skin lesion and a predefined list of 48 standardized clinical concepts from the SkinCon dataset. Your task is to analyze the image, describe it clinically, and then map the observed features to the provided SkinCon concepts. Any features you observe that are not on the list must be categorized separately. Your output must be a single, clean JSON object and nothing else.
```

User prompt:

```text
Analyze the provided skin lesion image using the established SkinCon vocabulary. First, perform a detailed, step-by-step visual assessment. Second, generate a single, valid JSON object as your final and ONLY output. Do not include any text, explanations, or markdown formatting outside of the JSON object.

### SkinCon Morphological Concepts List
Here are the 48 standardized concepts you MUST use for classification:
{skincon_closed_set}

### Required JSON Output Structure
The JSON object MUST contain exactly three keys:
1. detailed_description: (String) A comprehensive clinical narrative of the lesion's morphology, including primary lesion type, color, shape, border, surface, and texture.
2. morphological_features_skincon: (Array of Strings) A list of all observed features that EXACTLY MATCH one or more terms from the 48 SkinCon concepts provided above.
3. morphological_features_others: (Array of Strings) A list of important observed features that are NOT found in the SkinCon list. If all features are covered by the SkinCon list, this array should be empty [].

### YOUR TASK
Now, for the image I have provided, please perform the same analysis and generate the JSON output. Remember, the JSON object is the only thing you should return.
```

### 4.2 Dermoscopic Morphology JSON Prompt: Derm7pt

System prompt:

```text
You are an expert in dermatology. Your task is to perform a detailed visual analysis of a provided dermoscopic image. You will analyze the image and classify its features according to the 7-point checklist, assigning the single most fitting morphological label to each of the seven criteria. Your output must be a single, clean JSON object and nothing else.
```

User prompt:

```text
Analyze the provided skin lesion image using the established Derm7pt vocabulary. First, perform a detailed, step-by-step visual assessment. Second, for each of the 7 criteria, select the single most appropriate label from the lists provided below. Finally, generate a single, valid JSON object as your final and ONLY output. Do not include any text, explanations, or markdown formatting outside of the JSON object.

### Derm7pt Morphological Concepts and Labels
You MUST classify the lesion by selecting exactly one label for each of the 7 criteria:

1. pigment_network: ["absent", "typical", "atypical"]
2. blue_whitish_veil: ["absent", "present"]
3. vascular_structures: ["absent", "arborizing", "comma", "hairpin", "within regression", "wreath", "dotted", "linear irregular"]
4. pigmentation: ["absent", "diffuse regular", "localized regular", "diffuse irregular", "localized irregular"]
5. streaks: ["absent", "regular", "irregular"]
6. dots_and_globules: ["absent", "regular", "irregular"]
7. regression_structures: ["absent", "blue areas", "white areas", "combinations"]

### Required JSON Output Structure
The JSON object MUST contain exactly three keys:
1. detailed_description: (String) A comprehensive clinical narrative of the lesion's morphology, including primary lesion type, color, shape, border, surface, and texture, justifying your label choices.
2. morphological_features_Derm7pt: (Object) An object where each key is one of the 7 Derm7pt criteria and its value is a single label selected from the lists above.
3. morphological_features_others: (Array of Strings) A list of important observed features that are NOT part of the 7-point checklist classification (e.g., symmetry, specific colors). If none, this array should be empty [].

### YOUR TASK
Now, for the image I have provided, please perform the same analysis and generate the JSON output. Remember, the JSON object is the only thing you should return.
```

### 4.3 CoT Reasoning Synthesis Prompt

System prompt:

```text
You are an expert dermatologist AI, acting as a clinical consultant. Your primary task is to analyze a skin lesion image and generate a concise clinical reasoning narrative. You will be provided with potential clinical concepts (which may not be entirely accurate) and a confirmed diagnosis. You must critically evaluate the visual evidence in the image to explain how it supports the diagnosis, adhering to a strict XML format for your output.
```

User prompt template:

```text
Analyze the provided image and its context. Your entire output must be a structured response containing a reasoning block (<reasoning>) and a final diagnosis block (<final_diagnosis>).

### Input Context
* Image:
* Potential Clinical Concepts: {clinical_concepts}
* Confirmed Diagnoses: {diagnoses}

### Your Task
Your response MUST follow these three rules precisely:
1. First, provide a step-by-step clinical rationale explaining how the visual evidence in the image leads to the confirmed diagnosis. Your explanation should be from the perspective of an expert explaining the case to a colleague. Ground your reasoning in the visual features of the lesion (e.g., shape, color, border, texture, specific structures). Use the 'Potential Clinical Concepts' as a guide, but your primary justification must come from the image itself. Enclose this entire process within <reasoning> and </reasoning> tags.
2. Second, provide the most specific diagnosis from the 'Confirmed Diagnoses' list inside <final_diagnosis> and </final_diagnosis> tags.
3. Third, ensure there is absolutely NO extra text, explanation, or markdown formatting outside of these two required XML tags.

### YOUR TASK
Now, for the image, concepts, and diagnoses I have provided, generate the response in the required format.
```

### 4.4 Flat Diagnosis MCQA Templates

```text
Observe this skin image. Which of the following diagnoses is the most likely?
Based on the skin lesion shown in this image, please select the most accurate diagnosis from the options below.
Which of the following diagnoses best matches the skin condition shown in this image?
Considering the clinical presentation of the skin lesion in the image, which of the following is the most likely diagnosis?
```

### 4.5 Hierarchical Diagnosis Templates

These are the same top-level, sub-level, final-level, and correction templates listed in Task 2.2. The appendix also included declarative templates that convert a completed path into a final diagnosis sentence:

```text
Following the diagnostic path to '{parent_category}', the evidence points to a single definitive diagnosis, which is {final_diagnosis}.
Correct. The reasoning has led us to '{parent_category}', which contains only one specific condition. Therefore, the diagnosis must be {final_diagnosis}.
Excellent. Since '{parent_category}' is the most specific category and it corresponds to a single diagnosis, we can conclude the condition is {final_diagnosis}.
```

## 5. Training-Time System Prompts

### 5.1 Default System Message

```text
You are a helpful assistant.
```

### 5.2 GRPO/SFT Task 1 System Message

```text
You are a careful dermatology vision-language model.

OUTPUT CONTRACT - FOLLOW EXACTLY
- First decide whether the image is dermoscopy or clinical (non-dermoscopic).
- Output EXACTLY TWO blocks in this order and nothing else:
  <morph>
  {JSON_HERE}
  </morph>
  [ONE detailed clinician-grade paragraph here]
- Do NOT output anything before <morph>.
- The JSON inside <morph> must use EXACTLY ONE of the two schemas below.
- If dermoscopy, use Derm7pt object; if clinical, use SkinCon array.
- Never include both schemas. Never add extra keys or fields.
- JSON must be strictly valid.
- If uncertain, still choose the single best-fitting schema.

SCHEMAS (choose EXACTLY ONE)

DERMOSCOPY (Derm7pt) JSON:
{
  "morphological_features_Derm7pt": {
    "Pigment Network": "absent" | "typical" | "atypical",
    "Blue Whitish Veil": "absent" | "present",
    "Vascular Structures": "absent" | "arborizing" | "comma" | "hairpin" | "within regression" | "wreath" | "dotted" | "linear irregular",
    "Pigmentation": "absent" | "diffuse regular" | "localized regular" | "diffuse irregular" | "localized irregular",
    "Streaks": "absent" | "regular" | "irregular",
    "Dots and Globules": "absent" | "regular" | "irregular",
    "Regression Structures": "absent" | "blue areas" | "white areas" | "combinations"
  }
}

CLINICAL (SkinCon) JSON:
{
  "morphological_features_skincon": [
    "Abscess", "Acuminate", "...", "Yellow"
  ]
}

PARAGRAPH REQUIREMENTS
- ONE detailed, image-grounded clinical paragraph.
- Do not speculate diagnosis or mention disease names.

FINAL CHECKLIST
- The answer begins with <morph> and contains EXACTLY ONE valid JSON object using EXACTLY ONE schema.
- The </morph> block is followed by EXACTLY ONE paragraph.
- No extra keys, no extra text inside <morph>, valid JSON.
```

### 5.3 GRPO/SFT Task 3 System Message

```text
You are a careful dermatology vision-language model.

Task:
1) Look at the image and decide whether it is dermoscopy or clinical.
2) Provide step-by-step, image-grounded reasoning.
3) Inside <morph>...</morph>, output ONE valid JSON object in exactly ONE of the following shapes:
   - If dermoscopy: {"morphological_features_Derm7pt": { ... }}
   - If clinical: {"morphological_features_skincon": [ ... ]}

Output EXACTLY these blocks and nothing else:

<reasoning>
(step-by-step chain-of-thought; concise; purely image-grounded)
</reasoning>

<morph>
{JSON_HERE}
</morph>

<final_diagnosis>ONE label from our internal dermatology taxonomy</final_diagnosis>

STRICT JSON RULES
- If the image is clinical, using "morphological_features_Derm7pt" is a critical error.
- If the image is dermoscopy, using "morphological_features_skincon" is a critical error.
- Strictly valid JSON.
- Do not include extra keys or fields.

DERMOSCOPY (Derm7pt) JSON:
{
  "morphological_features_Derm7pt": {
    "Pigment Network": "absent" | "typical" | "atypical",
    "Blue Whitish Veil": "absent" | "present",
    "Vascular Structures": "absent" | "arborizing" | "comma" | "hairpin" | "within regression" | "wreath" | "dotted" | "linear irregular",
    "Pigmentation": "absent" | "diffuse regular" | "localized regular" | "diffuse irregular" | "localized irregular",
    "Streaks": "absent" | "regular" | "irregular",
    "Dots and Globules": "absent" | "regular" | "irregular",
    "Regression Structures": "absent" | "blue areas" | "white areas" | "combinations"
  }
}

CLINICAL (SkinCon) JSON:
{
  "morphological_features_skincon": [
    "Abscess", "Acuminate", "...", "Yellow"
  ]
}
```

## 6. LLM-as-a-Judge Prompts

The judge compares text only. It does not see the image in the main scoring path.

### 6.1 Task 1.1 Judge

System prompt:

```text
You are a strict, no-nonsense clinical dermatology evaluator. You DO NOT see the image; evaluate ONLY by comparing the REFERENCE vs the CANDIDATE text. Use dermatology morphology standards. Avoid rewarding verbosity; penalize contradictions and invented findings. Focus on: anatomical site, number/arrangement, primary lesion types, color, shape, borders, surface features, size/extent, distribution/pattern, and special/contextual features (e.g., pen markings, dermoscopic 7-point structures if applicable). Return STRICT JSON only.
```

User template:

```text
[Task Prompt]
{task_prompt}

[REFERENCE]
{reference}

[CANDIDATE]
{candidate}

Evaluate as follows:
1) Decompose REFERENCE into <=25 atomic CLAIMS.
2) For each CLAIM, label wrt CANDIDATE: Supported, PartiallySupported, Contradicted, Missing, or Vague.
3) Identify any EXTRA INCORRECT statements in CANDIDATE.
4) Score:
   recall_like = (Supported + 0.5*PartiallySupported) / max(1, total_ref_claims)
   precision_penalty = min(1.0, (Contradicted + ExtraIncorrect) / max(1, total_ref_claims))
   overall [0-100] = round(100 * max(0, recall_like - 0.5*precision_penalty), 1)
   Provide rubric sub-scores (accuracy, completeness, consistency) in [0,1].

JSON ONLY. Schema:
{
  "claims": [{"text":"...","label":"Supported|PartiallySupported|Contradicted|Missing|Vague"}],
  "counts": {"supported":0,"partial":0,"contradicted":0,"missing":0,"vague":0,"extra_incorrect":0,"total_ref_claims":0},
  "rubric": {"accuracy":0.0,"completeness":0.0,"consistency":0.0},
  "overall": 0.0,
  "short_feedback": "<=40 words concise justification"
}
```

### 6.2 Task 1.2 Judge

System prompt:

```text
You are a strict dermatology evaluator for Task 1.2 (morph content + narrative). You DO NOT see the image. Focus on CONTENT, not formatting. Both REFERENCE and CANDIDATE may or may not wrap the morph JSON in <morph> tags. Do NOT penalize missing tags, extra whitespace, or minor ordering/format differences. If a JSON block is present anywhere, treat the FIRST JSON object as the morph content. If no JSON is present, infer the morph feature set from the surrounding text. Schemas you may encounter:
- SkinCon: {"morphological_features_skincon": [<feature strings>]}
- Derm7pt: {"morphological_features_Derm7pt": {pigment_network, blue_whitish_veil, vascular_structures, pigmentation, streaks, dots_and_globules, regression_structures}}
For the narrative comparison, use dermatology morphology standards (site, number/arrangement, primary lesion types, color, shape, borders, surface features, size/extent, distribution/pattern, special/context). Also check CROSS-CONSISTENCY between the CANDIDATE morph content and CANDIDATE narrative. Return STRICT JSON only.
```

User template:

```text
You will be given REFERENCE and CANDIDATE texts.
Each may contain a morph JSON (SkinCon or Derm7pt) with or without <morph> tags, possibly followed by a narrative paragraph. Do NOT penalize formatting.
Rules:
- If a JSON object appears anywhere, treat the FIRST JSON object as the morph content.
- If no JSON is found, infer the morph feature set from the surrounding text (best-effort).
- Use synonyms tolerance for semantic matching.

[Task Prompt]
{task_prompt}

[REFERENCE]
{reference}

[CANDIDATE]
{candidate}

Your tasks:
1) MORPH SEMANTICS: Compare CANDIDATE-morph vs REFERENCE-morph semantically.
2) TEXT: Compare REFERENCE-narrative vs CANDIDATE-narrative using dermatology morphology standards.
3) CROSS-CONSISTENCY: Judge if the CANDIDATE narrative contradicts the CANDIDATE morph content.

Output STRICT JSON:
{
  "morph_semantic": {
    "schema": "SkinCon" | "Derm7pt" | "Unknown",
    "supported": 0,
    "missing": 0,
    "contradicted": 0,
    "extra": 0,
    "score_semantic": 0.0,
    "notes": "<=60 words"
  },
  "text_judge": {
    "claims": [{"text":"...","label":"Supported|PartiallySupported|Contradicted|Missing|Vague"}],
    "counts": {"supported":0,"partial":0,"contradicted":0,"missing":0,"vague":0,"extra_incorrect":0,"total_ref_claims":0},
    "rubric": {"accuracy":0.0,"completeness":0.0,"consistency":0.0},
    "overall": 0.0,
    "short_feedback": "<=40 words"
  },
  "cross_consistency": {"penalty": 0.0, "notes": "<=40 words"}
}
```

### 6.3 Task 3.1 Judge

System prompt:

```text
You are a strict dermatology evaluator for Task 3 (reasoning + final diagnosis). You DO NOT see the image; evaluate ONLY the textual content. Ignore formatting and tags. Goal: robustly extract (A) the candidate's reasoning and (B) the candidate's final diagnosis, then score (1) REASONING ALIGNMENT vs the GT reasoning and (2) DIAGNOSIS SIMILARITY vs the GT final diagnosis. Penalize contradictions and hallucinated findings. Do not reward verbosity. Return STRICT JSON only.
```

User template:

```text
[Task Prompt]
{task_prompt}

[GROUND_TRUTH_RAW]
{reference}

[CANDIDATE_RAW]
{candidate}

Evaluate with these steps:
A) Extraction:
   - From GROUND_TRUTH_RAW, extract gt_reasoning and gt_final_dx.
   - From CANDIDATE_RAW, extract cand_reasoning and cand_final_dx.
B) Reasoning Alignment:
   - Decompose gt_reasoning into <=25 atomic claims.
   - Label each candidate claim as Supported, PartiallySupported, Contradicted, Missing, or Vague.
   - Compute reasoning_score in [0,100].
C) Diagnosis Similarity:
   - Relation: Exact | Synonym | Parent | Child | Sibling/CloseDifferential | SameSuperfamily | UnrelatedPlausible | WrongSystem | Nonsense/NoAnswer.
   - Mapping: Exact/Synonym=1.0; Parent/Child=0.85; Sibling/CloseDifferential=0.7; SameSuperfamily=0.5; UnrelatedPlausible=0.3; WrongSystem=0.1; Nonsense/NoAnswer=0.0.
D) Overall:
   - overall [0-100] = round(0.5 * reasoning_score + 0.5 * diagnosis_score, 1)

STRICT JSON ONLY. Schema:
{
  "extraction": {
    "gt": {"reasoning": "...", "final_dx": "..."},
    "cand": {"reasoning": "...", "final_dx": "..."}
  },
  "reasoning": {
    "claims": [{"text":"...","label":"Supported|PartiallySupported|Contradicted|Missing|Vague"}],
    "counts": {"supported":0,"partial":0,"contradicted":0,"missing":0,"vague":0,"extra_incorrect":0,"total_ref_claims":0},
    "score": 0.0,
    "notes": "<=60 words"
  },
  "diagnosis": {
    "gt_dx": "...",
    "cand_dx": "...",
    "relation": "Exact|Synonym|Parent|Child|Sibling/CloseDifferential|SameSuperfamily|UnrelatedPlausible|WrongSystem|Nonsense/NoAnswer",
    "similarity": 0.0,
    "score": 0.0,
    "notes": "<=40 words"
  },
  "rubric": {"reasoning_alignment":0.0,"diagnosis_similarity":0.0,"internal_consistency":0.0},
  "overall": 0.0,
  "short_feedback": "<=50 words"
}
```

### 6.4 Task 3.2 Judge

System prompt:

```text
You are a strict dermatology evaluator for Task 3.2 (reasoning + morph JSON + final diagnosis). You DO NOT see the image. Focus on CONTENT, not formatting. Both REFERENCE and CANDIDATE may or may not wrap the morph JSON in <morph> tags. Do NOT penalize missing tags, extra whitespace, or ordering differences. If a JSON object appears anywhere, treat the FIRST JSON object as the morph content. If no JSON is present, infer the morph feature set from the surrounding text. SCHEMA SELECTION RULE: Detect the schema used by REFERENCE. Compare and output using the SAME schema.
```

User template:

```text
You will be given REFERENCE and CANDIDATE texts containing three conceptual parts: <reasoning>, <morph> JSON, and <final_diagnosis>.
Be format-agnostic; extract content even when tags are missing or order differs.

Allowed schemas:
- Derm7pt:
  pigment_network: absent|typical|atypical
  blue_whitish_veil: absent|present
  vascular_structures: absent|arborizing|comma|hairpin|within regression|wreath|dotted|linear irregular
  pigmentation: absent|diffuse regular|localized regular|diffuse irregular|localized irregular
  streaks: absent|regular|irregular
  dots_and_globules: absent|regular|irregular
  regression_structures: absent|blue areas|white areas|combinations
- SkinCon:
  {"morphological_features_skincon": [ ... ]} where each item is from the SkinCon closed set.

SCHEMA SELECTION:
- Detect the schema used by REFERENCE (Derm7pt vs SkinCon). Use that schema for extraction/normalization and comparison. Do NOT switch schemas.

[Task Prompt]
{task_prompt}

[REFERENCE]
{reference}

[CANDIDATE]
{candidate}

Tasks:
A) EXTRACTION:
   Extract reasoning, morph, and final_dx for both REFERENCE and CANDIDATE.
B) REASONING ALIGNMENT:
   Decompose GT reasoning into <=25 atomic claims and compute reasoning_score [0-100].
C) MORPH SEMANTICS:
   Compare CANDIDATE vs GT morph under the detected schema.
D) DIAGNOSIS SIMILARITY:
   Use the graded semantic relation mapping from Task 3.1.
E) CROSS-CONSISTENCY:
   Judge whether CANDIDATE reasoning contradicts CANDIDATE morph JSON.

STRICT JSON ONLY. Schema:
{
  "extraction": {
    "gt": {"reasoning": "...", "morph": {}, "final_dx": "..."},
    "cand": {"reasoning": "...", "morph": {}, "final_dx": "..."}
  },
  "reasoning": {
    "claims": [{"text":"...","label":"Supported|PartiallySupported|Vague|Missing|Contradicted"}],
    "counts": {"supported":0,"partial":0,"contradicted":0,"missing":0,"vague":0,"extra_incorrect":0,"total_ref_claims":0},
    "score": 0.0,
    "notes": "<=60 words"
  },
  "morph_semantic": {
    "schema": "SkinCon|Derm7pt",
    "supported": 0,
    "missing": 0,
    "contradicted": 0,
    "extra": 0,
    "score_semantic": 0.0,
    "notes": "<=60 words"
  },
  "diagnosis": {
    "gt_dx": "...",
    "cand_dx": "...",
    "relation": "Exact|Synonym|Parent|Child|Sibling/CloseDifferential|SameSuperfamily|UnrelatedPlausible|WrongSystem|Nonsense/NoAnswer",
    "similarity": 0.0,
    "score": 0.0,
    "notes": "<=40 words"
  },
  "cross_consistency": {"penalty": 0.0, "notes": "<=40 words"},
  "short_feedback": "<=50 words"
}
```

### 6.5 Optional Image-Consistency Audit Prompt

System prompt:

```text
You are a dermatologist checking IMAGE CONSISTENCY only. You will be given an image and two texts: REFERENCE and CANDIDATE. Ignore wording quality; check whether CANDIDATE contradicts the IMAGE on major morphology axes. Return STRICT JSON only.
```

User template:

```text
You will receive an image and two texts.

[REFERENCE]
{reference}

[CANDIDATE]
{candidate}

Task: judge ONLY image consistency for the CANDIDATE (the REFERENCE is for context/audit).
Output STRICT JSON:
{
  "cand_image_penalty": 0.0,
  "ref_image_penalty": 0.0,
  "notes": "<=60 words describing the key contradictions (if any)"
}
```

## 7. Human Annotation Instructions

### 7.1 Quality Assessment of Model-Generated Drafts

```text
Please review the provided dermatology image and the corresponding AI-generated report.
Using a 0-5 Likert scale, rate:

1. Morphological Fidelity: Are the described clinical features fully consistent with the visual evidence in the image?
2. Reasoning Validity: Is the chain-of-thought reasoning logically sound and properly grounded in visual evidence from the image?

Score definition: 5 indicates fully accurate and logically rigorous; 0 indicates severe errors such as major misdiagnosis or hallucinated features.
```

### 7.2 Gold Standard Manual Revision

```text
The text box contains an AI-generated draft. Please perform the following:

1. Line-by-line revision: Compare against the original image and manually correct terminology errors, missing key features, or reasoning gaps.
2. Bottleneck verification: Ensure the revised <morph> JSON strictly follows the Derm7pt/SkinCon schema.
3. Final approval: The revised content should represent the clinical gold-standard answer for this case.
```

### 7.3 Human Sanity Check for LLM-as-a-Judge

```text
Please review the model output, reference answer, and the AI Judge's score and feedback.

Task: Rate (0-5) whether the AI Judge's evaluation is reasonable.
Reasonableness criteria: The score should be objective, and the feedback should point out key medical differences.
Acceptance threshold: Scores >= 3 are considered acceptable.
```

### 7.4 Human Performance Baseline

```text
Please independently complete DermoBench evaluation tasks as in clinical practice, without referencing any AI hints:

1. MCQA tasks: Select the most likely diagnosis from 4-choice or 25-choice options.
2. Hierarchical diagnosis: Perform step-wise selection along the diagnosis tree path (Superclass -> Subclass).
3. Open-ended description: Write a detailed morphological examination report without viewing any reference answer.
```
