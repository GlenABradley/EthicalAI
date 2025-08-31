import argparse
import numpy as np
import json
from pathlib import Path
from coherence.encoders.text_sbert import get_default_encoder
from ethicalai.types import AxisPack, Axis
from ethicalai.axes.calibrate import pick_thresholds

def load_pack(pack_id):
    art_dir = Path("artifacts")
    meta_path = art_dir / f"axis_pack:{pack_id}.meta.json"
    npz_path = art_dir / f"axis_pack:{pack_id}.npz"
    arrs = np.load(npz_path)
    axes = [Axis(name=k, vector=arrs[k], threshold=0.0, provenance={}) for k in arrs.files]
    meta = json.loads(meta_path.read_text()).get("meta", {})
    return AxisPack(id=pack_id, axes=axes, dim=axes[0].vector.shape[0], meta=meta)

def save_pack(pack):
    art_dir = Path("artifacts")
    np.savez_compressed(art_dir / f"axis_pack:{pack.id}.npz", **{a.name: a.vector for a in pack.axes})
    (art_dir / f"axis_pack:{pack.id}.meta.json").write_text(json.dumps({"meta": pack.meta}))

def load_dataset(path):
    with open(path, 'r') as f:
        for line in f:
            yield json.loads(line.strip())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pack', required=True)
    parser.add_argument('--dataset', required=True)
    args = parser.parse_args()
    pack = load_pack(args.pack)
    axis_name = Path(args.dataset).stem
    scores = {axis_name: []}
    enc = get_default_encoder()
    for item in load_dataset(args.dataset):
        text = item['text']
        label = item['label']
        X = enc.encode_tokens(text.split())
        for ax in pack.axes:
            if ax.name == axis_name:
                score = np.mean([np.dot(x, ax.vector) for x in X])
                scores[axis_name].append((score, label))
                break
    pack = pick_thresholds(pack, scores, fpr_max=0.05)
    save_pack(pack)
    print(f"Calibrated {axis_name} with threshold {next((a.threshold for a in pack.axes if a.name == axis_name), 'N/A')}")

if __name__ == "__main__":
    main()
