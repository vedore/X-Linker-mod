"""Script to apply the model PECOS-EL (Disease or Chemical) 
for inference in a given input""" 
from tqdm import tqdm
from src.python.xlinker.utils import (
    load_model,
    load_kb_info,
    process_pecos_preds
)


# ----------------------------------------------------------------------------
# Input
# ----------------------------------------------------------------------------
kb = "medic" # or "ctd_chemicals"
# check other models in the directory 'data/models/trained'
model_dir = "data/models/trained/disease_200_1ep" 

#Label Representation via Positive Instance Feature Aggregation (PIFA). 
#Check Page 6 of the article: https://arxiv.org/pdf/2010.05878
clustering = "pifa"

top_k = 5 # Number of top-k predictions to return

# Example of input entities
input_entities = ["Hypertension", "Diabetes", "Cancer"]


# ----------------------------------------------------------------------------
# Load and setup model to apply
# ----------------------------------------------------------------------------
custom_xtf, tfidf_model, cluster_chain = load_model(model_dir, clustering)
print("model loaded!")

# ----------------------------------------------------------------------------
# Load KB info
# ----------------------------------------------------------------------------
id_2_name, index_2_id, synonym_2_id_lower, name_2_id_lower, kb_names, kb_synonyms = (
    load_kb_info(kb, inference=True)
)
print("KB info loaded!")


# ----------------------------------------------------------------------------
# Apply model to input entities
# ----------------------------------------------------------------------------
input_entities_processed = [entity.lower() for entity in input_entities]

x_linker_preds = custom_xtf.predict(
    input_entities_processed, X_feat=tfidf_model.predict(input_entities_processed), only_topk=top_k
)

pbar = tqdm(total=len(input_entities_processed))

output = ''

for i, entity in enumerate(input_entities_processed):
    mention_preds = x_linker_preds[i, :]
    mention_output = process_pecos_preds(
        entity, mention_preds, index_2_id, top_k, inference=True, 
        label_2_name=id_2_name
        )

    output += f'Entity: {entity}\nOutput: {mention_output}\n--------\n'
    pbar.update(1)

pbar.close()

print(output)
