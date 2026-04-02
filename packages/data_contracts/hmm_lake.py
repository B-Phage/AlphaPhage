HMM_LAKE_SCHEMA = {
    "protein_name": "string",
    "organism_name": "string",
    "hmm_domain_name": "string",
    "annot": "string",
    "evalue": "float",
    "score": "float",
    "bias": "float",
    "length": "int",
    "top_hmm_domain_hit_on_prot": "int",
    "top_protein_match_for_hmm": "int",
    "best_dom_env_start": "int",
    "best_dom_env_end": "int",
    "best_dom_env_length": "int",
    "sequence": "string",  # column from protein file; ['sequence']
}

HMM_LAKE_PRIMARY_KEYS = ["protein_name", "organism_name", "hmm_domain_name"]
