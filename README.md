# Multimodal Trolley Problem Experiment

Code for a three-provider LLM moral-bias study modelled on the Moral Machine Experiment (Awad et al., 2018). The experiment probes whether large language models encode demographic preferences (race, gender, age) when making autonomous vehicle dilemma decisions, and whether those preferences manifest differently under text-only versus multimodal (face-image) conditions.

---

## Experimental design

Each scenario presents an AV brake-failure dilemma requiring a choice between two pedestrian groups. Scenarios are generated across:

- **Three LLMs**: Claude (`claude-sonnet-4-6`), OpenAI (`gpt-4.1`), Gemini (`gemini-2.5-flash`)
- **Two arms**: text arm (demographic labels in natural language) and image arm (FairFace face photographs)
- **Four demographic categories**: Race (6 pairwise comparisons), Gender, Age, Utilitarianism (group size only)
- **Three system roles** assigned randomly per scenario: `default` (AV algorithm), `expert` (moral philosopher), `normal` (ordinary person)

### Counterbalancing

Every scenario index produces a **base** and a **mirror** pair. In the mirror, the left/right positions of the two groups are swapped and the inaction/action framing inverts. This design controls independently for:

- **Position bias** — tendency to prefer whichever group appears first or on a given side
- **Inaction bias** — tendency to prefer outcomes that require no active intervention

Genuine demographic preferences are identified only when a model saves the *same demographic group* in both versions of a pair (paired analysis section of the report).

### Image arm pipeline

1. **Perception call** — the model classifies demographic attributes from face photographs; output is validated against ground-truth FairFace labels.
2. **Choice call** — runs only if perception is valid. Scenarios where perception fails are excluded from analysis.

Both stages use `temperature=0` for full reproducibility.

---

## File structure

```
multimodal_TP_test/
├── main.py                # CLI entry point
├── scenario_generator.py  # Shared constants, scenario generation, API helpers
├── text_arm.py            # Text arm runner
├── image_arm.py           # Image arm runner (two-stage perception + choice pipeline)
├── face_sampler.py        # Indexes and samples from the FairFace dataset
└── report.py              # Generates a self-contained HTML report
```

Output directories (created automatically, relative to the project root):

```
../results/CSVs/text/V9/    # Text arm CSV results
../results/CSVs/image/V9/   # Image arm CSV results
../results/HTMLs/V9/        # HTML reports
```

Dataset (not committed — see Setup):

```
../data/fairface/
├── combined_curated_fairface.csv   # Curated label file (val + elderly train subset)
├── val/                            # FairFace validation images
└── train/                          # FairFace training images (elderly supplement)
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download FairFace

The image arm uses the [FairFace dataset](https://github.com/joojs/fairface) (Kärkkäinen & Joo, 2021). Download the validation and training image archives and unpack them so that images sit at:

```
../data/fairface/val/
../data/fairface/train/
```

The label CSVs (`combined_curated_fairface.csv`, `curated_fairface_val.csv`, `curated_fairface_train_elderly.csv`) are not committed to this repository due to file size. Place them at `../data/fairface/` alongside the image directories.

### 3. Set API keys

```bash
export ANTHROPIC_API_KEY=your_key
export OPENAI_API_KEY=your_key
export GEMINI_API_KEY=your_key
```

---

## Running the experiment

```bash
# Full run — text arm (1000 scenarios × 2 variants × 3 providers = 6000 rows)
python3 main.py text

# Full run — image arm (same scenarios with face photographs)
python3 main.py image

# Quick test with 5 scenarios
python3 main.py text 5
python3 main.py image 5

# Generate HTML report from the latest run pair
python3 main.py report

# Generate report from specific CSV files
python3 main.py report path/to/text.csv path/to/image.csv
```

Run the text arm before the image arm to avoid concurrent rate-limit collisions across providers.

---

## Reproducibility

`SEED = 2` and `temperature = 0.0` throughout. Scenario structure is fully deterministic — identical inputs always produce identical responses.

Perception reliability was validated before the main run using repeated perception calls on the same image groups at `temperature=0`:

| Model | Fully consistent groups | Per-field agreement |
|---|---|---|
| Claude (`claude-sonnet-4-6`) | 90% | 94–100% |
| OpenAI (`gpt-4.1`) | 85% | 88–93% |
| Gemini (`gemini-2.5-flash`) | 88% | 95–100% |

Remaining variability across all models is concentrated in age classification. Race and gender are near-deterministic at `temperature=0`.

---

## Citations

**FairFace dataset**

> Kärkkäinen, K., & Joo, J. (2021). FairFace: Face attribute dataset for balanced race, gender, and age for bias measurement and mitigation. *Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)*.

**Moral Machine Experiment**

> Awad, E., Dsouza, S., Kim, R., Schulz, J., Henrich, J., Shariff, A., Bonnefon, J.-F., & Rahwan, I. (2018). The Moral Machine experiment. *Nature*, 563, 59–64.
