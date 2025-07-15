"""This module contains utility functions used in the project"""
import json
import os
import bconv
import pandas as pd
import numpy as np
import scipy as sp
import xml.etree.ElementTree as ET
import networkx as nx
import ast
from src.python.kbs import KnowledgeBase
from src.python.abrv import run_Ab3P, parse_Ab3P_output


# --------------------------------------------------------------------------
# Parse or convert datasets
# --------------------------------------------------------------------------
def parse_json(in_filepath):
    """
    Parse a JSON file.
    """

    with open(in_filepath, "r", encoding="utf-8") as in_file:
        in_data = json.load(in_file)
        in_file.close()

    return in_data


def parse_brat(data_dir, dataset=None, ent_types=None):
    """
    Parse the brat annotations and extract the text and annotations.
    """

    ann_dir = f"{data_dir}"
    filenames = os.listdir(ann_dir)

    annotations = {}

    for i, file in enumerate(filenames):
        doc_annotations = []

        if file[-3:] == "ann":
            file_id = file[:-4]
            data = open(f"{ann_dir}{file}", "r", encoding="utf-8")
            document = data.readlines()
            data.close()

            # Parse each annotation
            for line in document:
                annot_type = ""

                if dataset == "craft_ncbi_taxon":
                    start = line.split("\t")[1].split(" ")[1]
                    end = line.split("\t")[1].split(" ")[2]
                    annotation_text = line.split("\t")[2].strip("\n")
                    kb_id = line.split("\t")[1].split(" ")[0].strip("NCBITaxon:")

                else:

                    if annot_type in ent_types:
                        start = ""
                        end = ""
                        annotation_text = line.split("\t")[2].strip("\n")
                        kb_id = line.split("\t")[1].split(" ")[0]

                doc_annotations.append((start, end, annotation_text, kb_id))

        # Open the corresponding txt file and extract the text
        text = open(f"{data_dir}/{file_id}.txt", "r", encoding="utf-8").read()

        annotations[file_id] = {"text": text, "annotations": doc_annotations}

    return annotations


def parse_bioc_json(in_filepath, ent_types=[], dataset=None):
    """
    Parse a BioC JSON file and returns the annotations. The annotations are 
    the input for an Entity Linking evaluation tool
    """

    data = parse_json(in_filepath)

    annotations = {}

    for data in data["documents"]:
        doc_id = data["id"]
        doc_text = ""
        doc_annotations = []

        for i, passage in enumerate(data["passages"]):
            doc_text += passage["text"]

            if i == 0:
                if dataset != "ncbi_disease":
                    # Add space between title and text
                    doc_text += "\n"

            for annot in passage["annotations"]:
                annot_type = annot["infons"]["type"]

                if annot_type in ent_types:
                    start = annot["locations"][0]["offset"]
                    end = str(int(start) + int(annot["locations"][0]["length"]))
                    ent_text = annot["text"]
                    kb_id = annot["infons"].get(
                        "cui", annot["infons"].get("identifier", None)
                    )

                    doc_annotations.append((start, end, ent_text, kb_id))

        annotations[doc_id] = {"text": doc_text, "annotations": doc_annotations}

    return annotations


def parse_pubtator_file(dataset, filename, ent_types, parsed_data):
    """
    Parse a Pubtator file and returns a dictionary with the text and
    annotations.
    """

    with open(filename, "r", encoding="utf-8") as pubtator_file:
        data = pubtator_file.readlines()
        pubtator_file.close()
        text = ""

        for line in data:

            if "|t|" in line:
                text += line + "\n"
                doc_id = line.split("|t|")[0]

            elif "|a|" in line:
                line_data = line.split("|a|")
                text += line + "\n"
                parsed_data[doc_id] = {"text": text, "annotations": []}
                text = ""

            else:
                line_data = line.split("\t")

                if len(line_data) == 6:
                    annot_type = line_data[4]
                    doc_id = line_data[0]

                    if annot_type in ent_types or dataset == "med_mentions":

                        if dataset == "med_mentions":
                            annot_type = "BIO"

                        annot = (
                            line_data[1],
                            line_data[2],
                            line_data[3],
                            annot_type,
                            line_data[5].strip("\n"),
                        )

                        parsed_data[doc_id]["annotations"].append(annot)

    return parsed_data


def parse_pubtator(dataset, ent_types=[], evaluate=True):
    """
    Parse a dataset in the Pubtator format and return a dictionary containing
    document texts and their annotations.

    Parameters
    ----------
    dataset : str
        Name of the dataset to parse.
    ent_types : lst, optional
        Types of entities to include in the annotations 
        (e.g. ['Disease', 'Chemical'])
    evaluate : bool, optional
        If True, includes evaluation datasets. If False, includes training and
        development datasets as well (default is True).

    Returns
    -------
    dict
        A dictionary containing the text and annotations with the format:
        {doc_id: {"text": text, "annotations": [(start, end, entity, type, id)]}}
    """

    data_dir = f"data/datasets/{dataset}"

    if dataset == "bc5cdr":
        data_dir += "/CDR.Corpus.v010516"

        filenames = ["CDR_TestSet.PubTator.txt"]

        if not evaluate:
            filenames.append("CDR_DevelopmentSet.PubTator.txt")
            filenames.append("CDR_TrainingSet.PubTator.txt")

    elif dataset == "ncbi_disease":
        filenames = ["NCBItestset_corpus.txt"]

        if not evaluate:
            filenames.append("NCBIdevelopset_corpus.txt")
            filenames.append("NCBItrainset_corpus.txt")

        ent_types = ["Modifier", "SpecificDisease", "DiseaseClass"]

    elif dataset == "biored":
        filenames = ["Test.PubTator"]

        if not evaluate:
            filenames.append("Train.PubTator")
            filenames.append("Dev.PubTator")

    elif dataset == "med_mentions":
        filenames = ["corpus_pubtator.txt"]

    parsed_data = {}

    for filename in filenames:
        parsed_data = parse_pubtator_file(
            dataset, f"{data_dir}/{filename}", ent_types, parsed_data
        )

    return parsed_data


def output_json(out_filepath, out_data):
    """
    Output a JSON file.
    """

    with open(out_filepath, "w") as out_file:
        json.dump(out_data, out_file, indent=4)
        out_file.close()


def convert_bioc_xml_2_bioc_json(in_filepath, out_filepath):
    """
    Convert a BioC XML file to a BioC JSON file.
    """

    coll = bconv.load(in_filepath, fmt="bioc_xml", byte_offsets=False)
    bconv.dump(coll, out_filepath, fmt="bioc_json")


def convert_bioc_json_2_bioc_xml(in_filepath, out_filepath):
    """
    Convert a BioC JSON file to a BioC XML file.
    """

    coll = bconv.load(in_filepath, fmt="bioc_json", byte_offsets=False)
    bconv.dump(coll, out_filepath, fmt="bioc_xml")


def convert_pubtator_2_bioc_json(in_filepath, out_filepath):
    """
    Convert a PubTator file to a BioC JSON file.
    """

    coll = bconv.load(in_filepath, fmt="pubtator")  # , byte_offsets=False)
    bconv.dump(coll, out_filepath, fmt="bioc_json")


def convert_bioc_xml_2_pubtator(in_filepath, out_filepath):
    """
    Convert a BioC XML file to a PubTator file.
    """

    coll = bconv.load(in_filepath, fmt="bioc_xml", byte_offsets=False)
    bconv.dump(coll, out_filepath, fmt="pubtator")


def convert_brat_2_bioc_json(
    in_dir, out_dir, out_filename, dataset_name=None, ent_types=None):
    """
    Convert a dataset in the Brat format to the BioC JSON format.
    """

    data = parse_brat(in_dir, dataset=dataset_name, ent_types=ent_types)
    out_dict = {"source": "", "date": "", "key": "", "infons": "", "documents": []}

    for i, doc_id in enumerate(data.keys()):
        out_dict["documents"].append({"id": doc_id, "infons": "", "passages": []})

        pass_dict = {"text": "", "offset": 0, "infons": "", "annotations": []}
        pass_dict["text"] = data[doc_id]["text"]

        doc_annots = data[doc_id]["annotations"]

        for annot in doc_annots:
            text = annot[2]
            kb_id = annot[3]
            start = int(annot[0])

            try:
                end = int(annot[1])

            except ValueError:
                # Some annotations in the craft_ncbi_taxon dataset are discontinous
                # Example: ('19378', '19385;19398', 'fission ... yeast', '4894')
                end = int(annot[1].split(";")[1])
            length = end - start

            annot_dict = {
                "id": annot,
                "infons": {"type": ent_types[0], "identifier": kb_id},
                "text": text,
                "locations": [{"offset": start, "length": length}],
            }

            pass_dict["annotations"].append(annot_dict)

        out_dict["documents"][i]["passages"].append(pass_dict)

    output_json(f"{out_dir}{out_filename}.json", out_dict)


def output_pubtator(annotations, out_filepath):
    """
    Output a Pubtator file containing given annotations
    """

    output = ""

    for doc_id in annotations.keys():
        title = annotations[doc_id]["text"].split("\n")[0]
        abstract = annotations[doc_id]["text"].split("\n")[2]
        output += f"{title}\n{abstract}\n"

        for annot in annotations[doc_id]["annotations"]:
            output += f"{doc_id}\t{annot[0]}\t{annot[1]}\t{annot[2]}\t{annot[3]}\t{annot[4]}\n"

        output += "\n"

    with open(out_filepath, "w") as f1:
        f1.write(output)
        f1.close()


def parse_dataset(
    input_format, dataset=None, data_dir=None, ner_filepath=None, ent_types=None
):
    """Parses a given dataset into a dictionary of annotations"""

    ent_types = ent_types.split(";")

    if input_format == "bioc_json":
        annotations = parse_bioc_json(
            ner_filepath, ent_types=ent_types, dataset=dataset
        )

    elif input_format == "pubtator":
        annotations = parse_pubtator(ner_filepath, ent_types=ent_types)

    elif input_format == "brat":
        annotations = parse_brat(data_dir, ent_types=ent_types)

    return annotations


def prepare_nlm_chem():
    """
    Prepare the NLM-Chem dataset for Entity Linking evaluation. Convert
    the dataset into the BioC JSON format and then generates a merged file
    with all annotations. Create the file 'data/datasets/nlm_chem/test.json'.
    """

    data_dir = "data/datasets/nlm_chem"
    pmids_test = open(f"{data_dir}/pmcids_test.txt", "r", encoding="utf-8").readlines()
    pmids_test = [pmid.strip("\n") for pmid in pmids_test]

    out_dir = f"{data_dir}/bioc_json/"
    os.makedirs(out_dir, exist_ok=True)

    merged_json = {}

    for i, pmid in enumerate(pmids_test):
        convert_bioc_xml_2_bioc_json(
            f"{data_dir}/ALL/{pmid}_v1.xml", f"{out_dir}/{pmid}.json"
        )

        # Merge all Json files into one
        data = parse_json(f"{out_dir}/{pmid}.json")

        if i == 0:
            merged_json = data

        elif i > 0:
            merged_json["documents"].extend(data["documents"])

    # Output merged json dict
    with open(f"{data_dir}/test.json", "w", encoding="utf-8") as out_file:
        json.dump(merged_json, out_file)
        out_file.close()


def convert_nlm_chem_2_pubtator():
    """
    Convert the NLM-Chem dataset to the Pubtator format. Create a single
    file containing all annotations.
    """

    data_dir = "data/datasets/nlm_chem"

    pmids_test = open(f"{data_dir}/pmcids_test.txt", "r", encoding="utf-8").readlines()
    pmids_test = [pmid.strip("\n") for pmid in pmids_test]

    out_dir = f"{data_dir}/pubtator/"
    os.makedirs(out_dir, exist_ok=True)

    for i, pmid in enumerate(pmids_test):
        convert_bioc_xml_2_pubtator(
            f"{data_dir}/ALL/{pmid}_v1.xml", f"{out_dir}/{pmid}"
        )

    # Combine all text files into a single pubtator file
    pubtator_files = os.listdir(out_dir)
    pubtator_files = [f"{out_dir}/{f}" for f in pubtator_files]

    with open(f"{data_dir}/test_pubtator.txt", "w", encoding="utf-8") as out_file:
        for f in pubtator_files:
            with open(f, "r", encoding="utf-8") as in_file:
                out_file.write(in_file.read())
                out_file.write("\n")


def convert_linnaeus_2_bioc_json():
    """Convert the Linnaeus dataset to the BioC JSON format.
    Create the file 'data/datasets/linnaeus/test.json'.
    """

    data_dir = "data/datasets/linnaeus"

    # Import annotations from "tags.tsv"
    annotations = pd.read_csv(
        f"{data_dir}/tags.tsv",
        sep="\t",
        names=["species id", "document", "start", "end", "text", "code"],
    )
    annotations_dict = {}

    for index in annotations.index:
        doc_id = annotations["document"][index]
        kb_id = annotations["species id"][index].strip("species:ncbi:")
        start = annotations["start"][index]
        end = annotations["end"][index]
        text = annotations["text"][index]
        key = f"{text}_{start}"

        if doc_id in annotations_dict.keys():
            annotations_dict[doc_id]["annotations"][key] = (start, end, text, kb_id)

        else:
            annotations_dict[doc_id] = {"annotations": {key: (start, end, text, kb_id)}}

    # Import text from "txt" files
    out_dict = {"source": "", "date": "", "key": "", "infons": "", "documents": []}
    filenames = os.listdir(f"{data_dir}/txt")

    for i, filename in enumerate(filenames):
        doc_id = filename.strip(".txt")
        out_dict["documents"].append({"id": doc_id, "infons": "", "passages": []})
        text = open(
            f"{data_dir}/txt/{filename}", "r", encoding="utf-8"
        ).read()  # .replace("\n", "$")

        # Add annotations to out_dict
        if doc_id in annotations_dict.keys():
            pass_dict = {"text": "", "offset": 0, "infons": "", "annotations": []}
            pass_dict["text"] = text

            doc_annots = annotations_dict[doc_id]["annotations"]

            for annot in doc_annots:
                text = doc_annots[annot][2]
                kb_id = doc_annots[annot][3]
                start = int(doc_annots[annot][0])
                end = int(doc_annots[annot][1])
                length = end - start

                annot_dict = {
                    "id": annot,
                    "infons": {"type": "Taxon", "identifier": kb_id},
                    "text": text,
                    "locations": [{"offset": start, "length": length}],
                }

                pass_dict["annotations"].append(annot_dict)

            out_dict["documents"][i]["passages"].append(pass_dict)

    output_json(f"{data_dir}/test.json", out_dict)


def convert_bc2gn_to_json():
    """
    Convert the BC2GN dataset to the BioC JSON format.
    Create the file 'data/datasets/bc2gn/test.json'.
    """

    data_dir = "data/datasets/bc2gn/bc2GNtest"

    # Import annotations
    with open(f"{data_dir}/bc2GNtest.genelist", "r") as f1:
        data = f1.readlines()
        f1.close()

    annotations = {}

    for annot in data:
        doc_id = annot.split("\t")[0]
        text = annot.split("\t")[2].strip("\n")
        kb_id = annot.split("\t")[1]

        if doc_id not in annotations.keys():
            annotations[doc_id] = {text: kb_id}

        else:
            annotations[doc_id][text] = kb_id

    txt_filenames = os.listdir(f"{data_dir}/bc2GNtestdocs")
    out_dict = {"source": "", "date": "", "key": "", "infons": "", "documents": []}

    for i, filename in enumerate(txt_filenames):
        doc_id = filename.strip(".txt")
        out_dict["documents"].append({"id": doc_id, "infons": "", "passages": []})

        if doc_id in annotations.keys():
            text = open(f"{data_dir}/bc2GNtestdocs/{filename}", "r").read()
            pass_dict = {"text": "", "offset": 0, "infons": "", "annotations": []}
            pass_dict["text"] = text

            doc_annots = annotations[doc_id]

            for annot in doc_annots:
                kb_id = doc_annots[annot]

                # Find the start and end
                start = text.find(annot)
                length = len(annot)

                annot_dict = {
                    "id": f"{annot}_{start}",
                    "infons": {"type": "Gene", "identifier": kb_id},
                    "text": annot,
                    "locations": [{"offset": start, "length": length}],
                }

                pass_dict["annotations"].append(annot_dict)

            out_dict["documents"][i]["passages"].append(pass_dict)

    out_dir = data_dir.strip("bc2GNtest")
    output_json(f"{out_dir}/test.json", out_dict)


def convert_datasets_to_bioc_json():
    """
    Convert the evaluations datasets into the BioCreative JSON format.
    In each dataset directory, the function generates a file named 'test.json'.
    Only needed if using the official evaluation script from BioCreative. 
    """

    print("Converting datasets to BioC JSON format...")

    # linnaeus:
    convert_linnaeus_2_bioc_json()
    print("Linnaeus dataset converted to BioC JSON format.")

    # nlm_chem:
    prepare_nlm_chem()
    print("NLM-Chem dataset converted to BioC JSON format.")

    # "bc5cdr":
    convert_bioc_xml_2_bioc_json(
        "data/datasets/bc5cdr/CDR.Corpus.v010516/CDR_TestSet.BioC.xml",
        "data/datasets/bc5cdr/test.json",
    )
    print("BC5CDR dataset converted to BioC JSON format.")

    # ncbi_disease:
    convert_pubtator_2_bioc_json(
        "data/datasets/ncbi_disease/NCBItestset_corpus.txt",
        "data/datasets/ncbi_disease/test_tmp.json",
    )
    # Change all entity types to 'Disease'
    os.system(
        """sed 's/"type": "Modifier"/"type": "Disease"/g' data/datasets/ncbi_disease/test_tmp.json | sed 's/"type": "SpecificDisease"/"type": "Disease"/g' | sed 's/"type": "DiseaseClass"/"type": "Disease"/g' > data/datasets/ncbi_disease/test_tmp2.json"""
    )
    os.system(
        """sed 's/"cui":/"identifier":/g' data/datasets/ncbi_disease/test_tmp2.json > data/datasets/ncbi_disease/test.json"""
    )
    os.system("rm data/datasets/ncbi_disease/test_tmp.json")
    os.system("rm data/datasets/ncbi_disease/test_tmp2.json")

    print("NCBI Disease dataset converted to BioC JSON format.")

    # med_mentions:
    # Change the every annotation type to 'BIO'
    os.system(
        """awk -F'\t' '{ print $1 "\t" $2 "\t" $3 "\t" $4 "\t" "BIO" "\t" $6 }' data/datasets/med_mentions/corpus_pubtator.txt > data/datasets/med_mentions/corpus_pubtator_type_corrected"""
    )

    annotations = parse_pubtator("med_mentions")
    output_pubtator(
        annotations, "data/datasets/med_mentions/corpus_pubtator_type_corrected.txt"
    )
    convert_pubtator_2_bioc_json(
        "data/datasets/med_mentions/corpus_pubtator_type_corrected.txt",
        "data/datasets/med_mentions/test.json",
    )
    print("MedMentions dataset converted to BioC JSON format.")

    # craft_ncbi_taxon:
    os.system("./src/bash/convert_craft_corpus.sh")
    convert_brat_2_bioc_json(
        "data/datasets/craft/concept-annotation/NCBITaxon/NCBITaxon/brat/",
        "data/datasets/craft/",
        "test_species",
        dataset_name="craft_ncbi_taxon",
        ent_types=["Taxon"],
    )
    print("CRAFT Corpus dataset converted to BioC JSON format.")

    # bc2gn:
    convert_bc2gn_to_json()
    print("BC2GN dataset converted to BioC JSON format.")


def generate_txt_files(
    input_format, dataset=None, data_dir=None, ner_filepath=None, ent_types=None
):
    """
    Generate text files for each document in given dataset.
    """

    annotations = parse_dataset(
        input_format,
        dataset=dataset,
        data_dir=data_dir,
        ner_filepath=ner_filepath,
        ent_types=ent_types,
    )

    out_dir = f"{data_dir}/txt"
    os.makedirs(out_dir, exist_ok=True)

    for doc_id in annotations.keys():
        with open(f"{out_dir}/{doc_id}.txt", "w", encoding="utf-8") as out_file:
            out_file.write(annotations[doc_id]["text"])
            out_file.close()


def dataset_stats():
    """
    Get stats for an evaluation dataset, such as the number of annotations
    with ID in the KB.
    """
    dataset = "linnaeus"
    kb = "ncbi_taxon"
    ent_type = "Taxon"

    annotations = parse_dataset(
        "bioc_json",
        dataset=None,
        data_dir=None,
        ner_filepath=f"data/datasets/{dataset}/test.json",
        ent_types=ent_type,
    )

    kb_ids = ""
    non_ids = ""
    label_2_name = parse_json(f"data/kbs/{kb}/label_2_name.json")

    for doc in annotations.keys():
        annots = annotations[doc]

        for annot in annots["annotations"]:

            kb_id = annot[3]
            kb_ids += f"{kb_id}\n"

            if kb_id not in label_2_name.keys():
                non_ids += f"{kb_id}\n"

    with open(f"stats/{dataset}_{kb}_{ent_type}_non_ids", "w") as out_file:
        out_file.write(non_ids)
        out_file.close()

    with open(f"stats/{dataset}_{kb}_{ent_type}_kb_ids", "w") as out_file:
        out_file.write(kb_ids)
        out_file.close()


# ---------------------------------------------------------------------------
# Retrieve the document ids from the evaluation datasets to exclude from
# training data
# ---------------------------------------------------------------------------
def get_pmid_from_pmcid(pmcid, email=None):
    """
    Get the PMID of an article from a given PMC ID.
    """

    Entrez.email = email  # Set your email

    try:
        handle = Entrez.read(Entrez.elink(dbfrom="pmc", db="pubmed", id=pmcid))
        links = handle[0]["LinkSetDb"][0]["Link"]
        for link in links:
            if link["Id"]:
                return link["Id"]
    except Exception as e:
        print("An error occurred:", e)
    return None


def get_linnaeus_pmids(datasets_dir, email=None):
    """
    Return a list of PMIDs from the LINNAEUS dataset.
    """

    filenames = os.listdir(f"{datasets_dir}/linnaeus/txt")
    pmcids = [filename.strip(".txt") for filename in filenames]
    pmcids = [filename.strip("pmcA") for filename in pmcids]

    # Convert PMCIDs to PMIDs
    pmids = []

    for pmcid in pmcids:
        pmid = get_pmid_from_pmcid(pmcid, email=email)
        if pmid:
            pmids.append(pmid)

    return pmids


def get_bc2gn_pmids(datasets_dir):
    """
    Return a list of PMIDs from the BC2GN dataset.
    """

    # Get PMIDS from test set
    test_filepath = f"{datasets_dir}/bc2gn/bc2GNtest/bc2GNtest.genelist"
    pmids = []

    with open(test_filepath, "r", encoding="utf-8") as in_file:
        for line in in_file.readlines():
            pmid = line.split("\t")[0]

            if pmid not in pmids:
                pmids.append(pmid)

        in_file.close()

    return pmids


def create_datasets_pmids_list(email=None):
    """Create a file with PMIDs to exclude from the training data since they 
    correspond to documents present in evaluation datasets. 
    The goal is to further generate unbiased training data.

    List of evaluation datasets:
    -MedMentions [x] -> UMLS
    -BC5CDR [x] -> MESH
    -NCBI Disease [x] -> MESH
    -BioRED [x] -> MESH
    -NLM-Chem [x] -> MESH
    -LINNAEUS [x] -> NCBI Taxonomy
    -BC2GN [x] -> NCBI Gene
    -CRAFT Corpus [x] -> several ontologies
    """
    datasets_dir = "data/datasets"
    pmids_list = []
    out_info = ""

    # Import MedMentions PMIDs
    # medmentions_filepath = f"{datasets_dir}/med_mentions/corpus_pubtator_pmids_all.txt"

    # with open(medmentions_filepath, "r", encoding="utf-8") as in_file:
    #    med_pmids = in_file.readlines()
    #    in_file.close()
    #    med_pmids_up = [pmid.strip("\n") for pmid in med_pmids]
    #    pmids_list.extend(med_pmids_up)
    #    out_info += f"MedMentions IDs added: {len(med_pmids_up)}\n"

    # Import NLM-Chem PMIDs
    nlmchem_filepath = f"{datasets_dir}/nlm_chem/pmcids_corpus.txt"

    with open(nlmchem_filepath, "r", encoding="utf-8") as in_file2:
        nlmchem_pmids = in_file2.readlines()
        in_file2.close()
        nlmchem_pmids_up = [pmid.strip("\n") for pmid in nlmchem_pmids]
        pmids_list.extend(nlmchem_pmids_up)
        out_info += f"NLM_Chem IDs added: {len(nlmchem_pmids_up)}\n"

    # Import CRAFT Corpus PMIDs
    # craft_filepath = f"{datasets_dir}/craft/articles/ids/craft-pmids.txt"

    # with open(craft_filepath, "r", encoding="utf-8") as in_file3:
    #    craft_pmids = in_file3.readlines()
    #    in_file3.close()

    #    craft_pmids_up = [pmid.strip("\n") for pmid in craft_pmids]
    #    pmids_list.extend(craft_pmids_up)
    #    out_info += f"CRAFT Corpus IDs added: {len(craft_pmids_up)}\n"

    # BC5CDR
    bc5cdr_annots = parse_pubtator(
        "bc5cdr", ent_types=["Disease", "Chemical"], evaluate=True
    )
    bc5cdr_pmids = list(bc5cdr_annots.keys())
    pmids_list.extend(bc5cdr_pmids)
    out_info += f"BC5CDR IDs added: {len(bc5cdr_pmids)}\n"

    # Import PMIDs from NCBI Disease
    ncbi_disease_annots = parse_pubtator(
        "ncbi_disease", ent_types=["Disease"], evaluate=True
    )
    ncbi_disease_pmids = list(ncbi_disease_annots.keys())
    pmids_list.extend(ncbi_disease_pmids)
    out_info += f"NCBI Disease IDs added: {len(ncbi_disease_pmids)}\n"

    # Import PMIDs from BioRED
    biored_annots = parse_pubtator(
        "biored", ent_types=["Disease", "Chemical"], evaluate=True
    )
    biored_pmids = list(biored_annots.keys())
    pmids_list.extend(biored_pmids)
    out_info += f"BioRED IDs added: {len(biored_pmids)}\n"

    # Import PMIDs from BC2GN
    # bc2gn_pmids = get_bc2gn_pmids(datasets_dir)
    # pmids_list.extend(bc2gn_pmids)
    # out_info += f"BC2GN IDs added: {len(bc2gn_pmids)}\n"

    # Import PMIDs from LINNAEUS
    # linnaeus_pmids_filename = f"{datasets_dir}/linnaeus/pmids_list.txt"

    # if not os.path.exists(linnaeus_pmids_filename):
    #    print("LINNAEUS PMIDs file not found. Generating it...")
    #    linnaeus_pmids = get_linnaeus_pmids(datasets_dir, email=email)
    # Create a file to store the pmids
    #    with open(linnaeus_pmids_filename, "w", encoding="utf-8") as linnaeus_file:
    #        for pmid in linnaeus_pmids:
    #            linnaeus_file.write(f"{pmid}\n")

    #        linnaeus_file.close()

    # else:
    #    with open(linnaeus_pmids_filename, "r", encoding="utf-8") as in_file4:
    #        linnaeus_pmids = in_file4.readlines()
    #        in_file4.close()
    #        linnaeus_pmids = [pmid.strip("\n") for pmid in linnaeus_pmids]

    # pmids_list.extend(linnaeus_pmids)
    # out_info += f"LINNAEUS IDs added: {len(linnaeus_pmids)}\n"

    # Remove duplicates
    pmids_list_dedup = [pmid.strip("\n").strip("_PMID") for pmid in pmids_list]
    pmids_list_dedup = list(set(pmids_list_dedup))
    out_info += f"Total number of PMIDs (deduplicated): {len(pmids_list_dedup)}"

    # Write PMIDs to file
    with open(f"{datasets_dir}/ignore_pmids.txt", "w", encoding="utf-8") as out_file:
        for pmid in pmids_list_dedup:
            out_file.write(f"{pmid}\n")

        out_file.close()

    # Output info file
    with open(
        f"{datasets_dir}/ignore_pmids_info.txt", "w", encoding="utf-8"
    ) as out_file2:
        out_file2.write(out_info)
        out_file2.close()


# -----------------------------------------------------------------------------
# Utils to deal with KB files
# -----------------------------------------------------------------------------
def generate_ncbi_taxon():
    """
    Generate a .tsv file including names and synonyms of the NCBI Taxonomy.
    """

    in_filename = "data/kbs/ncbi_taxon/tmp.csv"
    out_filename = "data/kbs/ncbi_taxon/ncbi_taxon.tsv"
    tmp_dict = {}

    with open(in_filename, "r") as in_file:
        data = in_file.readlines()

        for line in data:
            taxon_id, name = line.strip().split(",")[:2]

            if taxon_id in tmp_dict:
                tmp_dict[taxon_id].append(name)

            else:
                tmp_dict[taxon_id] = [name]

    output = "TaxonID\tTaxonName\tSyonyms\n"

    for taxon_id in tmp_dict:
        names = tmp_dict[taxon_id]
        syonyms = names[1:]
        syn_str = "|".join(syonyms)
        output += f"{taxon_id}\t{names[0]}\t{syn_str}\n"

    with open(out_filename, "w") as out_file:
        out_file.write(output)
        out_file.close()


def generate_kb_mappings(kb):
    """
    Generate mappings between knowledge base concept names and idenfitiers 
    (i.e. labels). The mappings are stored in JSON files.
    """

    data_dir = f"data/kbs/{kb}"

    if kb == "medic":
        data_filepath = f"{data_dir}/CTD_diseases.tsv"
        id_column = "DiseaseID"
        name_column = "DiseaseName"
        col_names = [
            "DiseaseName",
            "DiseaseID",
            "CasRN",
            "Definition",
            "ParentIDs",
            "TreeNumbers",
            "ParentTreeNumbers",
            "Synonyms",
            "pad",
        ]
        data = pd.read_csv(f"{data_filepath}", names=col_names, sep="\t", skiprows=29)

    elif kb == "ctd_chemicals":
        data_filepath = f"{data_dir}/CTD_chemicals.tsv"
        id_column = "ChemicalID"
        name_column = "ChemicalName"
        col_names = [
            "ChemicalName",
            "ChemicalID",
            "CasRN",
            "Definition",
            "ParentIDs",
            "TreeNumbers",
            "ParentTreeNumbers",
            "Synonyms",
            "pad",
        ]
        data = pd.read_csv(f"{data_filepath}", names=col_names, sep="\t", skiprows=29)

    elif kb == "ctd_genes":
        data_filepath = f"{data_dir}/CTD_genes.tsv"
        id_column = "GeneID"
        name_column = "GeneSymbol"
        col_names = [
            "GeneSymbol",
            "GeneName",
            "GeneID",
            "AltGeneIDs",
            "Synonyms",
            "BioGRIDIDs",
            "PharmGKBIDs",
            "UniProtIDs",
        ]
        data = pd.read_csv(f"{data_filepath}", names=col_names, sep="\t", skiprows=29)

    elif kb == "ncbi_taxon":
        data_filepath = f"{data_dir}/ncbi_taxon.tsv"
        id_column = "TaxonID"
        name_column = "TaxonName"
        col_names = ["TaxonID", "TaxonName", "Synonyms"]
        data = pd.read_csv(f"{data_filepath}", names=col_names, sep="\t", skiprows=1)

    data[id_column] = data[id_column]

    if kb == "medic" or kb == "ctd_chemicals":
        # Remove the prefix 'MESH:' from the identifiers
        data[id_column] = data[id_column].str.replace("MESH:", "")

    labels = data[id_column].tolist()
    ids_cleaned = labels
    names = data[name_column].tolist()
    synonym_2_id = {}
    id_2_synonym = {}
    label_2_def = {}

    for i, row in data.iterrows():

        # Build synonym_2_label
        if type(row["Synonyms"]) == str:
            synonyms = row["Synonyms"].split("|")

            for synonym in synonyms:
                synonym_2_id[synonym] = row[id_column]
                id_2_synonym[row[id_column]] = synonyms

        # Build definitions file
        if kb != "ctd_genes" and kb != "ncbi_taxon":
            definition = row["Definition"]

            if type(definition) == str:
                label_2_def[row[id_column]] = definition

    index_2_label = {}
    label_2_index = {}

    # Generate labels.txt
    with open(f"{data_dir}/labels.txt", "w", encoding="utf-8") as file1:
        for i, label in enumerate(labels):
            file1.write(f"{label}\n")
            index_2_label[i] = label
            label_2_index[label] = i
        file1.close()

    # Generate label_2_def.json
    with open(f"{data_dir}/label_2_def.json", "w", encoding="utf-8") as file2:
        json.dump(label_2_def, file2)
        file2.close()

    # Generate index_2_label.json and label_2_index.json
    with open(f"{data_dir}/index_2_label.json", "w", encoding="utf-8") as file3:
        json.dump(index_2_label, file3)
        file3.close()

    with open(f"{data_dir}/label_2_index.json", "w", encoding="utf-8") as file4:
        json.dump(label_2_index, file4)
        file4.close()

    # Create a dictionary from the DataFrame
    id_2_name = dict(zip(ids_cleaned, names))

    with open(f"{data_dir}/label_2_name.json", "w", encoding="utf-8") as file5:
        json.dump(id_2_name, file5)
        file5.close()

    # Generate name_2_label.json
    name_2_id = dict(zip(names, ids_cleaned))

    with open(f"{data_dir}/name_2_label.json", "w", encoding="utf-8") as file6:
        json.dump(name_2_id, file6)
        file6.close()

    # Generate synonym_2_label.json
    with open(f"{data_dir}/synonym_2_label.json", "w", encoding="utf-8") as file7:
        json.dump(synonym_2_id, file7)
        file7.close()

    # Generate label_2_synonym.json
    with open(f"{data_dir}/label_2_synonym.json", "w", encoding="utf-8") as file8:
        json.dump(id_2_synonym, file8)
        file8.close()


# -----------------------------------------------------------------------------
# Utils for X-Linker application
# -----------------------------------------------------------------------------
def add_predicted_kb_identifers(
    ner_filepath, predictions, ent_type, remove_prefix=False
):
    """
    Add predicted knowledge base identifiers to a file containing entity 
    annotations (i.e. corresponding to the output of the named entity 
    recognition stage.

    This function reads the entity annotations from a JSON file, adds the 
    predicted identifiers for each entity and generates a dictionary including
    the update annotations.

    Parameters
    ----------
    ner_filepath : str
        The path to the NER output file.
    predictions : dict
        A dictionary containing the predictions, where the format is:
        predictions[doc_id][entity] = (answer_kb_id, entity_type).
    ent_type : str
        The type of entities to process.
    remove_prefix : bool, optional
        If True, removes a specific prefix from the entity identifiers 
        (default is False).

    Returns
    -------
    out_dict
        The function updates the annotations dictionary with the predicted 
        identifiers.
    """

    ner_data = {}

    if ner_filepath[-4:] == "json" or ner_filepath[-4:] == "JSON":
        ner_data = parse_json(ner_filepath)

    # add predicted identifiers to NER output
    keys_to_keep = ["source", "date", "key", "infons"]
    out_dict = {key: ner_data[key] for key in ner_data if key in keys_to_keep}

    updated_documents = []

    for doc in ner_data["documents"]:
        doc_predictions = {}

        if doc["id"] in predictions.keys():
            doc_predictions = predictions[doc["id"]]

        keys_to_keep_2 = ["id", "infons", "relations"]
        doc_dict = {key: doc[key] for key in doc if key in keys_to_keep_2}
        updated_passages = []

        annot_counter = 0

        for passage in doc["passages"]:

            keys_to_keep_3 = ["infons", "offset", "text", "sentences", "relations"]
            passage_dict = {
                key: passage[key] for key in passage if key in keys_to_keep_3
            }

            passage_annots = []

            for annot in passage["annotations"]:
                keys_to_keep_4 = ["id", "text", "locations"]
                annot_dict = {key: annot[key] for key in annot if key in keys_to_keep_4}

                if ent_type == annot["infons"]["type"]:
                    search_key = str(annot_counter)

                    if search_key in doc_predictions.keys():
                        pred_kb_id = str(doc_predictions[search_key][0])

                        if "MESH:" in pred_kb_id and remove_prefix:
                            pred_kb_id = pred_kb_id.split("MESH:")[1]

                        annot_dict["infons"] = {"identifier": pred_kb_id}
                        annot_counter += 1

                    else:
                        annot_dict["infons"] = {"identifier": "None"}

                else:
                    annot_dict["infons"] = {"identifier": "None"}

                annot_dict["infons"]["type"] = annot["infons"]["type"]

                passage_annots.append(annot_dict)

            passage_dict["annotations"] = passage_annots
            updated_passages.append(passage_dict)

        doc_dict["passages"] = updated_passages
        updated_documents.append(doc_dict)
        out_dict["documents"] = updated_documents

    return out_dict


def prepare_input(test_annots_raw, abbreviations, id_2_name):
    """
    Prepare the input for the Entity Linking model (X-Linker or SapBERT). 
    The function filters out the annotations corresponding to NIL ids or IDs 
    that are not present in the used version of the target KB and lowercase
    each entity text.

    Parameters
    ----------
    test_annots_raw : list
        A list containing the raw annotations from the evaluation dataset. 
        Each element corresponds to an annotation in the string format.
    abbreviations : dict
        A dictionary containing the abbreviations for each document.
    id_2_name : dict
        A dictionary containing the mapping between knowledge base identifiers 
        and names.

    Returns
    -------
    test_input : list
        A list containing the input text for the Entity Linking model.
    test_annots : list
        A list containing the annotations to be linked.
    """

    # Filter out the annotations corresponding to NIL ids or
    # IDs that are not present in the current version of the target KB
    removed_annotations = []
    test_input = []
    test_annots = []

    for annot in test_annots_raw:
        doc_id = annot.split("\t")[0]
        annot_start = int(annot.split("\t")[1])
        annot_end = int(annot.split("\t")[2])
        annot_text = annot.split("\t")[3]
        annot_kb_id = annot.split("\t")[5].strip("\n").replace("MESH:", "")

        if (
            (annot_kb_id == "-1" or annot_kb_id not in id_2_name)
            and "|" not in annot_kb_id
            and "," not in annot_kb_id
        ):
            removed_annotations.append(annot)

        else:
            input_text = annot_text

            if doc_id in abbreviations:

                if annot_text in abbreviations[doc_id]:
                    input_text = abbreviations[doc_id][annot_text]

            test_input.append(input_text.lower())
            test_annots.append(
                [doc_id, annot_start, annot_end, annot_text, annot_kb_id]
            )

    assert len(test_input) == len(test_annots)
    print(f"{len(test_annots)} test instances loaded!")
    print(f"{len(removed_annotations)} annotations removed!")

    return test_input, test_annots


def get_dataset_abbreviations(dataset):
    """
    Get the abbreviations for a given dataset. If the abbreviations have not
    been detected yet, the function runs the Ab3P tool to detect them.

    Parameters
    ----------
    dataset : str
        The name of the dataset.

    Returns
    -------
    abbreviations : dict
        A dictionary containing the abbreviations for each document.
    """

    dataset_dir = f"data/datasets/{dataset}"
    abbrv_dir = f"{dataset_dir}/abbrv"
    abbrv_filepaths = []

    if os.path.isdir(abbrv_dir):
        abbrv_filepaths = os.listdir(abbrv_dir)

    else:
        os.makedirs(abbrv_dir, exist_ok=True)

    if len(abbrv_filepaths) == 0:
        print("Running Ab3P to detect abbreviations...")
        abbreviations = run_Ab3P(dataset_dir)

    elif len(abbrv_filepaths) > 0:
        print("Parsing abbreviations...")
        abbreviations = parse_Ab3P_output(f"{dataset_dir}/abbrv")

    return abbreviations


def calculate_topk_accuracy(df, topk_values):
    """
    Calculate the Top-k accuracy for each value of k in topk_values.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain 'code' and 'codes' columns.
        - 'code' is the true label
        - 'codes' is a list or a stringified list of predicted codes

    topk_values : list of int
        Values of k to compute Top-k accuracy for.

    Returns
    -------
    dict
        {k: accuracy} for each k in topk_values.
    """
    topk_accuracies = {k: 0 for k in topk_values}

    for _, row in df.iterrows():
        true_code = row["code"]
        predicted_codes = row["codes"]

        # Handle stringified list
        if isinstance(predicted_codes, str):
            try:
                predicted_codes = ast.literal_eval(predicted_codes)
            except (ValueError, SyntaxError):
                predicted_codes = [predicted_codes]

        if not isinstance(predicted_codes, list):
            predicted_codes = [predicted_codes]

        # Remove duplicates while preserving order
        seen = set()
        unique_candidates = [x for x in predicted_codes if not (x in seen or seen.add(x))]

        for k in topk_values:
            if true_code in unique_candidates[:k]:
                topk_accuracies[k] += 1

    total = len(df)
    return {k: correct / total for k, correct in topk_accuracies.items()}
