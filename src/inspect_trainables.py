# src/inspect_trainables.py
import os, argparse, tensorflow as tf
from models import build_lstm_model, build_transformer_model, build_tcn_model, build_scratch_3d_tcn_model

def build_model(name, num_classes, seq_len, img_size, use_pretrained=True):
    weights = "imagenet" if use_pretrained else None
    if name == "lstm":         return build_lstm_model(num_classes, seq_len, img_size, weights=weights)
    if name == "transformer":  return build_transformer_model(num_classes, seq_len, img_size, weights=weights)
    if name == "tcn":          return build_tcn_model(num_classes, seq_len, img_size, weights=weights)  # your code defaults to ImageNet
    if name == "scratch3d_tcn":return build_scratch_3d_tcn_model(num_classes, seq_len, img_size)
    raise ValueError(name)

def count_trainables(model):
    tot_w = sum(int(w.trainable) for w in model.weights)
    layers_trainable = sum(1 for l in model.layers if l.trainable)
    return tot_w, layers_trainable

def list_backbone_layers(model):
    # try to find mobilenetv2 subgraph by name
    hits = [l for l in model.layers if 'mobilenetv2' in l.name.lower()]
    return hits

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["lstm","transformer","tcn","scratch3d_tcn"], required=True)
    ap.add_argument("--seq_len", type=int, default=16)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--num_classes", type=int, default=4)
    ap.add_argument("--pretrained", action="store_true", help="show ImageNet usage")
    args = ap.parse_args()

    mdl = build_model(args.model, args.num_classes, args.seq_len, args.img_size,
                      use_pretrained=(args.model!="scratch3d_tcn"))
    print(f"[INFO] Built model: {mdl.name}")
    tw, tl = count_trainables(mdl)
    print(f"[INFO] BEFORE FREEZE: trainable weights={tw}, trainable layers={tl}")
    bb_layers = list_backbone_layers(mdl)
    print(f"[INFO] Backbone layers detected: {len(bb_layers)} (e.g., {[l.name for l in bb_layers[:5]]})")

    # simulate your warm-up freeze loop EFFECT (for demonstration only)
    for l in mdl.layers:
        if isinstance(l, tf.keras.layers.BatchNormalization):
            l.trainable = False
        elif hasattr(l, "name") and ("mobilenetv2" in l.name.lower()
              or l.__class__.__name__ in ("Conv2D","DepthwiseConv2D")):
            l.trainable = False

    tw2, tl2 = count_trainables(mdl)
    print(f"[INFO] AFTER WARM-UP FREEZE: trainable weights={tw2}, trainable layers={tl2}")

    # simulate your fine-tune partial unfreeze (top 30% conv-like)
    conv_like = [l for l in mdl.layers if l.__class__.__name__ in ("Conv2D","DepthwiseConv2D")]
    cutoff = int(0.7 * len(conv_like))
    for l in conv_like[cutoff:]:
        l.trainable = True

    tw3, tl3 = count_trainables(mdl)
    print(f"[INFO] AFTER PARTIAL FT UNFREEZE: trainable weights={tw3}, trainable layers={tl3}")
