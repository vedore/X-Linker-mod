"""Train the PECOS-EL Disease or Chemical model"""
import argparse
import copy
import os
import logging
import torch

import time
import numpy as np

# import wandb
from logging.handlers import RotatingFileHandler

from xmr4el.featurization.preprocessor import Preprocessor
from xmr4el.xmr.skeleton_builder import SkeletonBuilder

# wandb.login()

# ------------------------------------------------------------
# Check the available GPUs
# ------------------------------------------------------------
# See https://discuss.pytorch.org/t/solved-make-sure-that-pytorch-using-gpu-to-compute/4870/6
# torch.cuda.is_available() 


# ------------------------------------------------------------
# Args
# ------------------------------------------------------------
parser = argparse.ArgumentParser(description="Train XR-Transformer model")
parser.add_argument("-run_name", type=str)
parser.add_argument("-ent_type", type=str, default="Disease", help="")
parser.add_argument("-kb", type=str, default="medic", help="")
parser.add_argument(
    "-model",
    type=str,
    default="bert",
    choices=["bert", "roberta", "biobert", "scibert", "pubmedbert"],
    help="",
)
parser.add_argument(
    "-clustering", type=str, default="pifa", choices=["pifa", "pifa_lf"], help=""
)
parser.add_argument("-epochs", type=int, default=10, help="")
parser.add_argument("-batch_size", type=int, default=32, help="")
parser.add_argument("--only_kb", action="store_true", help="")
parser.add_argument("--max_inst", type=int, help="")
parser.add_argument("--batch_gen_workers", type=int, help="")
args = parser.parse_args()

# ------------------------------------------------------------
# Filepaths
# ------------------------------------------------------------
DATA_DIR = "data/train"
KB_DIR = f"data/kbs/{args.kb}"
RUN_NAME = args.run_name

model_dir = f"data/models/trained/{RUN_NAME}"
os.makedirs(model_dir, exist_ok=True)

# ------------------------------------------------------------
# Configure the logger
# ------------------------------------------------------------
log_format = "%(asctime)s - %(levelname)s - %(message)s"

logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        logging.StreamHandler(),  # To write to console
        RotatingFileHandler(
            filename=f"log/TRAIN_{RUN_NAME}.log",
            maxBytes=5 * 1024 * 1024,
            backupCount=2,
        ),  # To write to a rotating file
    ],
)
logging.info("\n------------------------------------------")

logging.info(f"CUDA is available:{torch.cuda.is_available()}")
logging.info(f"CUDA DEVICE COUNT:{torch.cuda.device_count()}")

# ------------------------------------------------------------
# Parse training data
# ------------------------------------------------------------
logging.info("Parsing training data")

if args.only_kb:
    train_filepath = f"{DATA_DIR}/{args.ent_type}/labels.txt"

else:
    if args.ent_type == "Disease":
        train_filepath = f"{DATA_DIR}/Disease/train_Disease_{args.max_inst}.txt"

    elif args.ent_type == "Chemical":
        train_filepath = f"{DATA_DIR}/Chemical/train_Chemical.txt"

labels_filepath = f"{KB_DIR}/labels.txt"

parsed_train_data = Preprocessor().load_data_labels_from_file(
    train_filepath, labels_filepath
)

logging.info(f"Parse train file: {train_filepath}")

Y_train = parsed_train_data["labels_matrix"]
raw_labels = parsed_train_data["raw_labels"]
X_train = parsed_train_data["corpus"]
x_cross_train = parsed_train_data["cross_corpus"]
label_enconder = parsed_train_data["label_encoder"]

# Use training label frequency scores as costs -> build relevance matrix
R_train = copy.deepcopy(Y_train)

logging.info(
    f"Constructed training corpus len={len(X_train)}, training label matrix with shape={Y_train.shape} and nnz={Y_train.nnz}"
)

# ------------------------------------------------------------
# Feature extraction: build TF-IDF model with training corpus
# ------------------------------------------------------------
# vectorizer_config = {
#     "type": "tfidf",
#     "kwargs": {
#         "base_vect_configs": [
#             {
#                 "ngram_range": [1, 2],
#                 "max_df_ratio": 0.98,
#                 "analyzer": "word",
#                 "buffer_size": 0,
#                 "threads": 30,
#             },
#         ],
#     },
# }

# if os.path.exists(tf_idf_filepath):
#     logging.info("Loading TF-IDF model from disk")
#     tfidf_model = xmr4el.Preprocessor.load(tf_idf_filepath)

# else:
#     logging.info("Training TF-IDF model")
#     tfidf_model = Preprocessor.train(X_train, vectorizer_config)
#     tfidf_model.save(tf_idf_filepath)
#     logging.info("Saved TF-IDF model")


# X_train_feat = tfidf_model.predict(X_train)
# logging.info(
#     f"Constructed TRAINING feature matrix with shape={X_train_feat.shape} and nnz={X_train_feat.nnz}"
# )

# del tfidf_model

# ------------------------------------------------------------
# Construct label hierarchy
# see https://github.com/amzn/pecos/blob/2283da828b9ce6061ca2125ff62d8a37c934550f/pecos/xmc/base.py#L1827
# -----------------------------------------------------------------
# logging.info(f"Building cluster chain with method {args.clustering}")

# cluster_chain_filepath = f"{model_dir}/cluster_chain_{args.clustering}"

# Z_filepath = None

# if args.clustering == "pifa_lf":
#     Z_filepath = f"data/kbs/{args.kb}/Z_{args.kb}_300_dim_20.npz"

# cluster_chain = get_cluster_chain(
#     X=X_train,
#     X_feat=X_train_feat,
#     Y=Y_train,
#     method=args.clustering,
#     cluster_chain_filepath=cluster_chain_filepath,
#     Z_filepath=Z_filepath,
# )

# ------------------------------------------------------------
# Train XR-Transformer model
# ------------------------------------------------------------
logging.info("Training model")

start = time.time()

min_leaf_size = 30
depth = 10
n_features = 768
max_n_clusters = 30
min_n_clusters = 6

vectorizer_config = {
    "type": "tfidf", 
    "kwargs": {"max_features": 20000}
}
    
transformer_config = {
    "type": "sentencetsapbert",
    "kwargs": {"batch_size": 400}
    }

clustering_config = {
    "type": "sklearnminibatchkmeans",
    "kwargs": {
        "n_clusters": 8,  # This should be determined by your tuning process
        "init": "k-means++",
        "max_iter": 500,  # Increased from 300
        "batch_size": 0,  # Larger batch size for more stable updates
        "verbose": 0,
        "compute_labels": True,
        "random_state": 42,  # Fixed for reproducibility
        "tol": 1e-4,  # Added small tolerance for early stopping
        "max_no_improvement": 20,  # More patience for improvement
        "init_size": 24,  # 3 * n_clusters (3*8=24)
        "n_init": 5,  # Run multiple initializations, pick best
        "reassignment_ratio": 0.01,
    }
}

# clustering_config = {
#     "type": "faisskmeans",  # Matches the registered name in your ClusterMeta system
#     "kwargs": {
#         "n_clusters": 8,           # Default cluster count (will be overridden by tuner)
#         "max_iter": 300,           # Max iterations per run
#         "nredo": 1,               # Number of initializations (FAISS calls this nredo)
#         "gpu": True,               # Enable GPU acceleration
#         "verbose": False,          # Disable progress prints
#         "spherical": False,        # Set True for cosine similarity (L2 normalizes first)
#         "seed": 42,                # Random seed (FAISS uses this for centroid init)
#         "tol": 1e-4,               # Early stopping tolerance
#     }
# }

classifier_config = {
    "type": "sklearnlogisticregression",
    "kwargs": {
        "n_jobs": -1, 
        "random_state": 0,
        "penalty":"l2",           
        "C": 1.0,               
        "solver":"lbfgs",    
        "max_iter":1000
        },
}

# classifier_config = {
#     "type": "lightgbmclassifier",
#     "kwargs": {
#         "objective": "multiclass",
#         "boosting_type": "gbdt",
#         "learning_rate": 0.05,
#         "n_estimators": 300,
#         "num_leaves": 64,
#         "subsample": 0.8,
#         "colsample_bytree": 0.8,
#         "n_jobs": -1,
#         "random_state": 42
#     }
# }

# training_file = os.path.join(os.getcwd(), "test/test_data/train/disease/train_Disease_100.txt")

# trn_corpus = Preprocessor.load_data_from_file(train_filepath=training_file)

pipe = SkeletonBuilder(
    vectorizer_config,
    transformer_config, 
    clustering_config, 
    classifier_config, 
    n_features, 
    max_n_clusters, 
    min_n_clusters, 
    min_leaf_size, 
    depth, 
    dtype=np.float32
    )

htree = pipe.execute(
    raw_labels,
    x_cross_train,
    X_train,
    Y_train,
)

# Print the tree structure
print(htree)

# Save the tree
save_dir = os.path.join(os.getcwd(), "test/test_data/saved_trees")  # Ensure this path is correct and writable
htree.save(save_dir)

end = time.time()
print(f"{end - start} secs of running")

# start a new wandb run to track this script
# wandb.init(
# set the wandb project where this run will be logged
#    project=f"x_linker_{args.ent_type}",
#    name=RUN_NAME,
# track hyperparameters and run metadata
#    config=train_params
# )

# train_problem = MLProblemWithText(X_train, Y_train, X_feat=X_train_feat)

# custom_xtf = XTransformer.train(
#     train_problem,
#     R=R_train,
#     clustering=cluster_chain,
#     train_params=train_params,
#     verbose_level=3,
# )

logging.info("Training completed!")
# custom_xtf.save(f"{model_dir}/xtransformer")
