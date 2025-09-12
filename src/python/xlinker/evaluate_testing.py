"""Script to evaluate PECOS-EL and X-Linker in datasets"""
import time
import os
import numpy as np
from typing import Counter
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

from xmr4el.xmr.model import XModel

start = time.time()

def read_codes_file(filepath):
    code_lists = []

    with open(filepath, 'r') as f:
        for line in f:
            # Strip whitespace and split by '|'
            codes = line.strip().split('|')
            if codes:
                code_lists.append(codes)

    return code_lists

def filter_labels_and_inputs(gold_labels, input_texts, input_annots, allowed_labels):
    """
    Filters out gold_labels (list of lists) and corresponding input_texts
    where the first label in each gold label list is not in allowed_labels.

    Args:
        gold_labels (List[List[str]]): Nested list of gold labels.
        input_texts (List[str]): Raw input texts, aligned with gold_labels.
        allowed_labels (Iterable[str]): Set or list of valid labels.

    Returns:
        Tuple[List[List[str]], List[str]]: Filtered gold_labels and input_texts.
    """
    allowed_set = set(allowed_labels)

    filtered_labels = []
    filtered_annots = []
    filtered_texts = []

    for label_list, text, annot in zip(gold_labels, input_texts, input_annots):
        if label_list and label_list[0] in allowed_set:
            filtered_labels.append(label_list)
            filtered_texts.append(text)
            filtered_annots.append(annot)
            

    return filtered_labels, filtered_texts, filtered_annots

# Parse arguments
parser = ArgumentParser()
parser.add_argument("-dataset", type=str, required=True)
parser.add_argument("-ent_type", type=str, required=True)
parser.add_argument("-kb", type=str, required=True)
parser.add_argument("-model_dir", type=str, required=True)
parser.add_argument("-top_k", type=int, default=5)
parser.add_argument("-beam_size", type=int, default=5)
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
trained_xtree = XModel.load(args.model_dir)

print(len(trained_xtree.initial_labels))

# exit()

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

# print(test_input, len(test_input), "\n")

# print(test_annots, len(test_annots))

# exit()

print("Test instances loaded!")

# ----------------------------------------------------------------------------
# Apply model to test instances
# ----------------------------------------------------------------------------

code_lists = read_codes_file("test/test_data/labels_bc5cdr_disease_medic.txt")

# x_linker_preds = Predict.inference(trained_xtree, code_lists, test_input, k=args.top_k)

# print(trained_xtree)

gold_labels = read_codes_file("test/test_data/labels_bc5cdr_disease_medic.txt") # Need to filter out the ones that werent used.
    
filtered_labels, filtered_texts, filtered_annots = filter_labels_and_inputs(gold_labels, test_input, test_annots, trained_xtree.initial_labels)

counter_label_list_1 = 0
counter_label_list_2 = 0
for label_list in filtered_labels:
    if len(label_list) > 1:
        counter_label_list_2 += 1
        # print(label_list)
    else:
        counter_label_list_1 += 1

# print(counter_label_list_2, counter_label_list_1)
# print(filtered_labels)


# 10 Counter({0: 1264, 1: 21})
# 100 Counter({0: 1244, 1: 41}) Counter({1: 1285}) # Optimal classifier 1 job done
print(f"Beam Size: {args.beam_size}, TopK: {args.top_k}")
routes, scores = trained_xtree.predict(filtered_texts, beam_size=args.beam_size, topk=args.top_k, fusion="lp_fusion", topk_mode="global")
    
# print(score_matrix[0]["leaf_global_labels"])
    
trained_labels = np.array(trained_xtree.initial_labels)
    
# Get global label ids array from the score_matrix
# global_labels = score_matrix.global_labels  # shape (n_labels,)

hit_counts = []
for r in routes:
    qi = r["query_index"]
    # print(qi)
    # union of all labels reachable by the final surviving leaves
    cand = set()
    # for p in r.get("paths", []):
        # print("Leaf paths", p.get("leaf_global_labels"))
        # print("Leaf Scores", p.get("scores"), "\n")
    # print("Final path", r.get("final_path").get("leaf_global_labels", []), "\n")
    cand.update(trained_labels[r.get("final_path").get("leaf_global_labels", [])])
    gold = set(filtered_labels[qi])
    hit_counts.append(len(cand & gold))
        
print("Hit counts per query:", Counter(hit_counts))
print("Average hits:", np.mean(hit_counts))

end = time.time()
print(f"{end - start} secs of running")

# x_linker_preds = custom_xtf.predict(
#     test_input, X_feat=tfidf_model.predict(test_input), only_topk=args.top_k
# )

# [[1, 2, 3, 4, 5]]

# exit()

print("Linking test instances...")

output = []
pbar = tqdm(total=len(filtered_annots))

for i, annotation in enumerate(filtered_annots):
    mention_preds = scores[i, :]

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
    
    topk_accuracies = calculate_topk_accuracy(predictions_df, [1, 5, 10, 20, 50, 100, 200, 500])
    print(f"Top-k accuracies with PPR: {topk_accuracies}")

else:
    # Evaluate model performance
    pred_path = f"data/evaluation_{args.dataset}_{args.ent_type}.tsv"
    predictions_df.to_csv(pred_path, sep="\t", index=False)
    topk_accuracies = calculate_topk_accuracy(predictions_df, [1, 5, 10, 20, 50, 100, 200, 500])
    print(f"Top-k accuracies Without PPR: {topk_accuracies}")
    topK_list = [list(topk_accuracies.values())]
    df = pd.DataFrame(topK_list)
    df.to_csv("data.tsv", sep="\t", index=False)

end = time.time()
print(f"{end - start} secs of running")