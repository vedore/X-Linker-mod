"""Script to evaluate PECOS-EL and X-Linker in datasets"""
import time
import os
import pandas as pd
import src.python.xlinker.ppr as ppr

from argparse import ArgumentParser, BooleanOptionalAction
from src.python.xlinker.utils import (
    load_model,
    load_kb_info,
    process_pecos_preds,
    apply_pipeline_to_mention,
)

from src.python.utils import (
    get_dataset_abbreviations,
    prepare_input,
    calculate_topk_accuracy,
)

from tqdm import tqdm

from xmr4el.xmr.skeleton import Skeleton
from xmr4el.predict.predict import Predict

start = time.time()

# Parse arguments
parser = ArgumentParser()
parser.add_argument("-dataset", type=str, required=True)
parser.add_argument("-ent_type", type=str, required=True)
parser.add_argument("-kb", type=str, required=True)
parser.add_argument("-model_dir", type=str, required=True)
parser.add_argument("-top_k", type=int, default=5)
parser.add_argument("-clustering", type=str, default="pifa")
parser.add_argument("--abbrv", default=False, action=BooleanOptionalAction)
parser.add_argument("--pipeline", default=False, action=BooleanOptionalAction)
parser.add_argument("--threshold", type=float, default=0.1)
parser.add_argument("--ppr", default=False, action=BooleanOptionalAction)
parser.add_argument("--fuzzy_top_k", type=int, default=1)
parser.add_argument("--unseen", default=False, action=BooleanOptionalAction)
args = parser.parse_args()

# ----------------------------------------------------------------------------
# Load and setup model to apply
# ----------------------------------------------------------------------------
# custom_xtf, tfidf_model, cluster_chain = load_model(args.model_dir, args.clustering)
# print("model loaded!")
"""Load the tree"""

# train_disease_100
trained_xtree = Skeleton.load(args.model_dir)

# ----------------------------------------------------------------------------
# Load KB info
# ----------------------------------------------------------------------------
id_2_name, index_2_id, synonym_2_id_lower, name_2_id_lower, kb_names, kb_synonyms = (
    load_kb_info(args.kb, inference=True)
)
print("KB info loaded!")

# -------------------------------------------------------------------------------
# Get abbreviations in dataset
# -------------------------------------------------------------------------------
abbreviations = {}

if args.abbrv:
    abbreviations = get_dataset_abbreviations(args.dataset)
    print("Abbreviations loaded!")

# ----------------------------------------------------------------------------
# Import test instances
# ----------------------------------------------------------------------------
test_path = f"data/datasets/{args.dataset}/test_{args.ent_type}.txt"

if args.unseen:
    test_path = f"data/datasets/{args.dataset}/test_{args.ent_type}_unseen.txt"

with open(test_path, "r") as f:
    test_annots_raw = f.readlines()
    f.close()

test_input, test_annots = prepare_input(test_annots_raw, abbreviations, id_2_name)
print("Test instances loaded!")

# ----------------------------------------------------------------------------
# Apply model to test instances
# ----------------------------------------------------------------------------

transformer_config = {
    "type": "sentencetbiobert",
    "kwargs": {"batch_size": 400}
    }

encoder_config = {
    "type": "minilm_l6_v2",
    "kwargs": {}
}

"""
encoder_config = {
    "type": "biolinkbert",
    "kwargs": {}
}
"""


x_linker_preds = Predict.inference(trained_xtree, test_input, transformer_config, k=args.top_k)

# x_linker_preds = custom_xtf.predict(
#     test_input, X_feat=tfidf_model.predict(test_input), only_topk=args.top_k
# )

# [[1, 2, 3, 4, 5]]

print(x_linker_preds, x_linker_preds.shape)

print("Linking test instances...")

output = []
pbar = tqdm(total=len(test_annots))

for i, annotation in enumerate(test_annots):
    mention_preds = x_linker_preds[i, :]

    if args.pipeline:
        # Apply pipeline to every mention in test set
        mention_output = apply_pipeline_to_mention(
            test_input[i],
            annotation,
            mention_preds,
            kb_names,
            kb_synonyms,
            name_2_id_lower,
            synonym_2_id_lower,
            index_2_id,
            top_k=args.top_k,
            fuzzy_top_k=args.fuzzy_top_k,
            threshold=args.threshold,
        )

    else:
        # Just consider the X-linker predictions
        mention_output = process_pecos_preds(
            annotation, mention_preds, index_2_id, args.top_k
        )

    output.append(mention_output)
    pbar.update(1)
pbar.close()

# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------
# Convert predictions to DataFrame
predictions_df = pd.DataFrame(
    output, columns=["doc_id", "start", "end", "text", "code", "codes", "scores"]
)

if args.ppr:
    # Prepare input for PPR
    run_name = f"{args.dataset}_{args.ent_type}_{args.kb}"
    os.makedirs(f"data/REEL/{run_name}", exist_ok=True)
    pred_path = f"data/REEL/{run_name}/xlinker_preds.tsv"
    predictions_df.to_csv(pred_path, sep="\t", index=False)
    
    ppr.prepare_ppr_input(
        run_name,
        predictions_df,
        args.ent_type,
        fuzzy_top_k=args.fuzzy_top_k,
        kb=args.kb,
    )

    # Build the disambiguation graph, run PPR and process the results
    ppr.run(entity_type=args.ent_type, kb=args.kb, reel_dir=f"data/REEL/{run_name}")
    
    topk_accuracies = calculate_topk_accuracy(predictions_df, [1, 3, 5, 10])
    print(f"Top-k accuracies with PPR: {topk_accuracies}")

else:
    # Evaluate model performance
    pred_path = f"data/evaluation_{args.dataset}_{args.ent_type}.tsv"
    predictions_df.to_csv(pred_path, sep="\t", index=False)
    topk_accuracies = calculate_topk_accuracy(predictions_df, [1, 3, 5, 10])
    print(f"Top-k accuracies Without PPR: {topk_accuracies}")
    topK_list = [list(topk_accuracies.values())]
    df = pd.DataFrame(topK_list)
    df.to_csv("data.tsv", sep="\t", index=False)

end = time.time()
print(f"{end - start} secs of running")