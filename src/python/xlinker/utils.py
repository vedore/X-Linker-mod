"""Utility functions for PECOS-EL"""
import os
from pecos.xmc import Indexer, LabelEmbeddingFactory
from pecos.utils.cluster_util import ClusterChain
from pecos.utils import smat_util
from pecos.utils.featurization.text.preprocess import Preprocessor
from pecos.utils.cluster_util import ClusterChain
from pecos.xmc.xtransformer.model import XTransformer
from src.python.utils import parse_json, parse_dataset, add_predicted_kb_identifers
from src.python.kbs import KnowledgeBase
from src.python.candidates import map_to_kb, output_candidates_file



def get_cluster_chain(X=None, X_feat=None, Y=None, method="", 
                    cluster_chain_filepath="", Z_filepath=""):


    if os.path.exists(cluster_chain_filepath):
        cluster_chain = ClusterChain.load(cluster_chain_filepath)

    else:

        if method == "pifa":
            label_feat = LabelEmbeddingFactory.create(Y=Y, 
                                    X=X_feat, 
                                    method="pifa",
                                    threads=30, 
                                    normalized_Y=True)

        elif method == "pifa_lf":
            # METHOD: PIFA + LF ->  Create label embedding by 
            # concatenating pifa embeddings and provided existing 
            # label embedding    
            
            # # Load external KB/Label features/embeddings
            Z_feat = smat_util.load_matrix(Z_filepath)
            print(label_feat.shape)
            print(X_feat.shape, Y.shape, Z_feat.shape)
            
            label_feat = LabelEmbeddingFactory.create(Y=Y, 
                                                    X=X_feat, 
                                                    Z=Z_feat.toarray(), 
                                                    method="pifa_lf_concat",
                                                    threads=30, 
                                                    normalized_Y=True)

        cluster_chain = Indexer.gen(label_feat)
        cluster_chain.save(cluster_chain_filepath)
    
    return cluster_chain"


def load_model(model_dir, clustering_method="pifa"):
    """Loads a PECOS-EL model from disk"""
    custom_xtf = XTransformer.load(f"{model_dir}/xtransformer")
    tfidf_model = Preprocessor.load(f"{model_dir}/tfidf_model")
    cluster_chain = ClusterChain.load(f"{model_dir}/cluster_chain_{clustering_method}")

    return custom_xtf, tfidf_model, cluster_chain


def prepare_input(parsed_dataset, abbreviations=None):
    """Converts a parsed dataset text into the input format required by the 
    PECOS-EL model for prediction.

    Parameters
    ----------
    parsed_dataset : list of dict
        A list of dictionaries representing the parsed dataset. Each 
        dictionary contains information about a document, including its text
    abbreviations : dict of str to str, optional
        A dictionary of abbreviations mapping abbreviated terms to their full 
        forms. Default is None.

    Returns
    -------
    formatted_data : list of dict
        A list of dictionaries where each dictionary is formatted according to the
        requirements of the PECOS-EL model. 
    """

    window_size = 0
    test_input = {}
    
    for doc in parsed_dataset:
        doc_annots = parsed_dataset[doc]['annotations']
        test_input[doc] = []
        text = parsed_dataset[doc]['text']
        
        doc_abbrvs = {}
        
        if doc in abbreviations.keys():
           doc_abbrvs = abbreviations[doc] 
       
        for annot in doc_annots:
            annot_text = annot[2]

            if annot_text in doc_abbrvs.keys():
                annot_text = doc_abbrvs[annot_text]

            a_start = int(annot[0])
            a_end = int(annot[1]) 
            l_w_start = a_start - window_size
            r_w_end = a_end + window_size

            if l_w_start < 0:
                l_w_start = 0
            
            if r_w_end > len(text):
                r_w_end = len(text)

            tagged_text = f'{text[l_w_start:a_start]}@{annot_text}${text[a_end:r_w_end]}'

            if window_size > 0:
                tagged_text = f'{text[l_w_start:a_start]}@{annot_text}$ {text[a_end:r_w_end]}'
            
            test_input[doc].append(tagged_text)
    
    return test_input


def lower_dict_keys(input_dict):
    """Converts the keys of a dictionary to lowercase"""
    return {k.lower(): v for k, v in input_dict.items()}


def lower_list(input_list):
    """Converts a list of strings to lowercase"""
    return [item.lower() for item in input_list]


def load_kb_info(kb, inference=False):
    """Loads the information of a knowledge base from disk to be used in the 
    PECOS-EL model."""

    data_dir = f"data/kbs/{kb}"

    with open(f"{data_dir}/labels.txt", "r", encoding="utf-8") as fin:
        labels = [ll.strip() for ll in fin.readlines()]
        fin.close()

    # Open mappings label to name
    label_2_name = parse_json(f"{data_dir}/label_2_name.json")
    index_2_label = parse_json(f"{data_dir}/index_2_label.json")
    
    if inference:
        name_2_label = parse_json(f"{data_dir}/name_2_label.json")
        synonym_2_label = parse_json(f"{data_dir}/synonym_2_label.json")
        kb_names = lower_list(name_2_label.keys())
        kb_synonyms = lower_list(synonym_2_label.keys())
        name_2_label_lower = lower_dict_keys(name_2_label)
        synonym_2_label_lower = lower_dict_keys(synonym_2_label)

        return label_2_name, index_2_label, synonym_2_label_lower, name_2_label_lower, kb_names, kb_synonyms
    
    else:
        return labels, label_2_name, index_2_label


def load_test(dataset, ent_type, format, abbreviations=None):
    """Import given dataset, including document texts and respective 
    annotations"""
                   
    NER_FILEPATH = f"data/datasets/{dataset}/test.json"

    annotations = parse_dataset(
        format,
        data_dir=None,
        dataset=dataset,
        ner_filepath=NER_FILEPATH,
        ent_types=ent_type
        )
    
    test_input = prepare_input(annotations, abbreviations=abbreviations)

    return test_input, annotations, NER_FILEPATH


def load_kb_object(kb):
    """Load Knowledge base object"""

    return KnowledgeBase(kb=kb, input_format="tsv") 


def process_predictions(output, annotations, entity_type, labels=None, label_2_name=None):
    """Processes the predictions of the PECOS-EL model"""

    doc_out = {}
    
    for i in range(len(annotations)):
        pred_index = output[i, :].indices[0]# -1
        pred_label = labels[pred_index]
        key = str(i)
        doc_out[key] = (pred_label, entity_type)

    return doc_out


def retrieve_candidates_list(
    predictions=None,
    doc_annotations=None,
    index_2_label=None,
    label_2_name=None,
    kb_obj=None,
    top_k=None,
    threshold=0.10,
    ):
    """Process the predictions of the PECOS-EL model to generate a list of 
    the top K candidates for each annotation in the input."""
    doc_preds = {}

    for i, annot in enumerate(doc_annotations):
        start = annot[0]
        end = annot[1]
        ent_text = annot[2]
        doc_preds[str(i)] = []
        
        pred_indexes = predictions[i, :].indices.tolist()
        
        top_cand_score = predictions[i, :].data[0]
        top_cand_label = index_2_label[f"{pred_indexes[0]}"]
        top_cand_text = label_2_name[top_cand_label]
        search_key = str(i)
        doc_preds[search_key].append((start, end, top_cand_text, top_cand_label, top_cand_score))
       
        if top_cand_score < threshold:
            del doc_preds[search_key][0]
            
            #complete candidates list with candidates retrived by string matching
            matches = map_to_kb(ent_text, kb_obj.name_2_id, kb_obj.synonym_2_id, 1)

            for match in matches:
                doc_preds[search_key].append(
                    (start, end, match['name'], match['kb_id'], match['score']))

    return doc_preds


def prepare_output(pred_dict, ner_filepath, ent_type, dataset):
    """# Process Output and generate file"""

    remove_prefix = False

    if dataset == "bc5cdr" or dataset == "biored":
        remove_prefix = True
   
    out_dict = add_predicted_kb_identifers(
        ner_filepath, pred_dict, ent_type, remove_prefix=remove_prefix)
    
    return out_dict


def process_pecos_preds(annotation, mention_preds, index_2_label, top_k):

    # Add all X-Linker predictions
    pred_labels = []
    pred_scores = []   
    
    for k in range(top_k):
        pred_score = float(mention_preds.data[k])
        pred_scores.append(pred_score)
        pred_index = mention_preds.indices[k]
        pred_label = index_2_label[str(pred_index)]
        pred_labels.append(pred_label)

    return [
        annotation[0],
        annotation[1],
        annotation[2],
        annotation[3],
        annotation[4],
        pred_labels,
        pred_scores
    ]


def apply_pipeline_to_mention(
        input_text, 
        annotation,
        mention_preds, 
        kb_names,
        kb_synonyms,
        name_2_id,
        synonym_2_id,
        index_2_label,
        top_k=1,
        fuzzy_top_k=1,
        threshold=0.15
    ):
    """Applies X-Linker pipeline to given input mention"""
    
    output = []
    annot_text = annotation[3]
    true_label = annotation[4]
    #-----------------------------------------
    #   Get exact match from KB
    #-----------------------------------------
    kb_matches = map_to_kb(
                    input_text, 
                    kb_names,
                    kb_synonyms,
                    name_2_id,
                    synonym_2_id, 
                    top_k=fuzzy_top_k
                )

    #-----------------------------------------------
    # Process X-Linker predictions
    #-----------------------------------------------
    pecos_output = process_pecos_preds(annotation, mention_preds, index_2_label, top_k)
    labels_to_add, scores_to_add = [], []

        
    if kb_matches[0]['score'] == 1.0:
        labels_to_add.append(kb_matches[0]['kb_id'])
        scores_to_add.append(kb_matches[0]['score'])

        if pecos_output[6][0] == 1.0:
            labels_to_add.append(pecos_output[5][0])
            scores_to_add.append(pecos_output[6][0])

    else:

        if pecos_output[6][0] >= threshold:
            labels_to_add.append(pecos_output[5][0])
            scores_to_add.append(pecos_output[6][0])

        else:

            for i, label in enumerate(pecos_output[5]):
                labels_to_add.append(label)
                scores_to_add.append(pecos_output[6][i])

            for i, match in enumerate(kb_matches):
                labels_to_add.append(match['kb_id'])
                scores_to_add.append(match['score'])

    output = [
        annotation[0], 
        annotation[1], 
        annotation[2], 
        annotation[3], 
        annotation[4], 
        labels_to_add, 
        scores_to_add
    ]
    
    return output