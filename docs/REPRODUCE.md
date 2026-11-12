# Experiment commands

Run the commands from the repository root in the environment described in [the README](../README.md). They assume [the data layout and model locations](DATA.md). Use a clean runs/ directory for a new full reproduction. The reported results use seeds 0, 1, and 2 unless otherwise specified.

## Reference detector recipes

Use the manuscript's exact reference checkpoints for numerical comparisons where available. The configurations in configs/detectors/ specify the detector training arguments. The resulting model may differ across Ultralytics, PyTorch, or CUDA versions. The benchmark labels and metrics are detector-specific.

### XWOD YOLOv8m

Copy the dataset YAML to the XWOD root so that Ultralytics resolves split paths relative to the dataset. The recorded run used SGD, initial learning rate 0.001, seed 0, and automatic batch sizing.

    cp configs/detectors/xwod-dataset.yaml data/xwod/data_fixed.yaml
    yolo cfg=configs/detectors/xwod-yolov8m.yaml data="$PWD/data/xwod/data_fixed.yaml"

Use the resulting best.pt at runs/weatherfilm_runs/baseline_lr001/weights/best.pt. Then run the main XWOD commands in [the README](../README.md).

### BDD-IW

    python core/prep_bdd.py --test-frac 0.15 --seed 0
    python core/prep_bdd_yolo.py
    yolo cfg=configs/detectors/bdd-iw-yolov8m.yaml data="$PWD/data/bdd_yolo/bdd.yaml"
    python core/gen_understanding_labels_bdd.py --split all
    python core/gen_detectability_labels_bdd.py --split all
    python core/extract_patch_tokens_bdd.py

The BDD-IW detector recipe uses 200 epochs, patience 50 and batch 16. Use the best weights in runs/weatherfilm_runs/bdd_baseline-2/weights/best.pt. Train the BDD-IW readouts:

    for seed in 0 1 2; do
      python core/train_uwm_bdd.py --seed "$seed"
      python core/train_cnn_bdd.py --seed "$seed"
    done

The BDD training scripts save test metrics in their output directories under runs/weatherfilm_runs/. The split is image-random, with a held-out portion of the original training partition. The original validation partition is used for model selection.

## XWOD representation and layer comparisons

    for split in train val test; do
      python core/gen_text_embeddings.py --split "$split"
    done
    for seed in 0 1 2; do
      python core/train_text_baseline.py --seed "$seed"
      python core/train_cnn_baseline.py --seed "$seed"
      python core/train_vit_baseline.py --seed "$seed"
    done
    python core/eval_test_cnn.py --seeds 0 1 2
    python core/aggregate_seeds.py --seeds 0 1 2

For layer ablation, extract the L-2 and L-6 caches and reuse the main L-4 cache. Each layer configuration specifies its cache directory and training arguments.

    python core/extract_patch_tokens.py --layer -2 --output-dir data/layers/L-2
    python core/extract_patch_tokens.py --layer -6 --output-dir data/layers/L-6
    for layer in -2 -4 -6; do
      for seed in 0 1 2; do
        cfg="configs/B1_L${layer}_seed${seed}.json"
        python train.py --config "$cfg"
      done
    done

Evaluate a trained checkpoint with val.py, using the same config as training:

    python val.py --config configs/B1_L-2_seed1.json \
      --checkpoint runs/B1/manual_repeat_L-2_seed1/uwm_L-2_lam3_revision_seed1/best_uwm.pt \
      --device cuda

For a loss-weight sensitivity run, use core/train_understanding.py with --lam and a distinct --name for each setting. The manuscript default is 3. Use core/train_ablation_tasks.py for auxiliary-head ablations and inspect its --help for task selection.

## Targeted prompts and statistical-feature baseline

The targeted text generation uses the recorded validation-only batch plan, greedy generation and fixed prompt. GPU generation and text encoding run before training the lightweight text readout.

    for split in train val test; do
      python gpu_representation/generate_targeted_text.py --split "$split" \
        --out-dir runs/C1/targeted_embeddings
    done
    for seed in 0 1 2; do
      python train.py --config "configs/C1_seed${seed}.json"
    done
    python val_text_failure.py

The statistical-feature (SF) baseline pools spatial mean, maximum and standard deviation from the reference YOLOv8m backbone and uses the same hard-event definition as PaDU.

    python experiments/C2_statistical_features/extract.py
    for seed in 0 1 2; do
      python experiments/C2_statistical_features/train.py --seed "$seed" --out-dir "runs/C2/seed${seed}"
      python experiments/C2_statistical_features/val.py \
        --checkpoint "runs/C2/seed${seed}/best.pt" --output-dir "runs/C2/val_seed${seed}"
    done

This implements the statistical-feature comparison used for the manuscript with the shared detector and dataset. It is an adaptation of the cited SF method.

## Cross-detector evaluation

Complete the main XWOD run and export per-frame PaDU predictions first. Prepare the pinned local RT-DETRv2-R50 snapshot described in [Data and models](DATA.md). The first command scores the COCO-initialized detector with the fixed PaDU monitor. The second trains the distinct XWOD-tuned arm.

    python cross_detector_rtdetrv2.py
    python experiments/C3_rtdetrv2_r50/train.py
    for split in train val test; do
      python experiments/C3_rtdetrv2_r50/val.py \
        --checkpoint runs/C3/rtdetrv2_r50_finetuned_seed0/best.pt \
        --split "$split" \
        --output-dir "runs/C3/rtdetrv2_r50_finetuned_seed0/eval_${split}"
    done
    python summarize_cross_detector_details.py

The main cross-detector analysis keeps the original training thresholds. The sensitivity analysis estimates the second detector's tertiles from its training scores.

## Federated experiments

The original scripts run centralized comparisons, weather/density partitions, aggregation variants, quantization, client participation and simulated Byzantine updates. Start the original XWOD main comparison and the later participation and attack experiments separately:

    python run_all.py --only main
    python run_all2.py --only byz_majority_ms,byz_minority_ms,participation

The BDD-IW and random-versus-weather configurations select the corresponding checkpoint-producing driver.

    for seed in 0 1 2; do
      python train.py --config "configs/B2_bdd_seed${seed}.json"
      python train.py --config "configs/B3_weather_seed${seed}.json"
      python train.py --config "configs/B3_random_seed${seed}.json"
    done

The configuration runner writes training logs, validation-selected checkpoints and client partitions under runs/federation/. Use --out-dir to set the output directory for each repeat. For low-memory setups, inspect the original driver's --mode options before modifying the published execution recipe.

## Runtime measurements

With the main PaDU checkpoints, XWOD caches and inputs in place, use an otherwise idle GPU:

    python benchmark_pipeline_revision.py

After the targeted-text and SF runs, execute:

    python benchmark_remaining_pipelines.py

The first script records batch-one PaDU latency after ten warmup frames. The second adds the complete targeted-text and statistical-feature baselines. The runtime numbers depend on hardware, library versions, and filesystem cache behavior.

## Metric definitions

Seed summaries use population standard deviations. Average precision follows the noninterpolated definition. Frame-bootstrap samples are shared across PaDU seeds and condition on the fitted models. Cross-detector thresholds are fitted on training scores.

See [Data and models](DATA.md) for the evaluation split protocol.
