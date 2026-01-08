"""
SQLite base HMM to MSA functions
 Relies on existing SQLite DB tables with sequences and HMM annotations.
"""

import os
import sys

import pandas as pd

# import data_utils.sqlite3_utils as sql_utils
import pyhmmer
import utils.consol_log as log
from utils.duckdb_client import DuckDBClient

logger = log.get_logger(level=log.DEBUG)


def append_hmm_to_tbl(
    hmm_scan_df: pd.DataFrame, protein_py_easel_obj, db_path: str, sql_table_name: str
):
    """
    Append core hmm scan results to a sql table alongside the sequence string.
    If the gene did not get any HMM hits, the table to append will still contain
    the gene because an outer join of prodigal-gv genes and HMM hit results is used

    Inputs
        hmm_scan_df:    outputs of HMM scanning
        protein_py_easel_obj: block read from pyhmmer.easel.SequenceFile
        db_file:        path to SQLite DB
        sql_table_name: table in DB (new or existing from same process)
    """
    db = DuckDBClient(db_path)
    col_from_hmm_scan = [
        "protein_name",
        "organism_name",
        "hmm_domain_name",
        "annot",
        "evalue",
        "score",
        "bias",
        "length",
        "top_hmm_domain_hit_on_prot",
        "top_protein_match_for_hmm",
        "best_dom_env_start",
        "best_dom_env_end",
        "best_dom_env_length",
    ]
    # Plus columns from protein file; ['sequence']

    if not db.table_exists(sql_table_name):
        logger.warning(f"Creating {sql_table_name}")
        db.con.execute(
            f"""
        CREATE TABLE {sql_table_name} (
            protein_name VARCHAR PRIMARY KEY,
            organism_name VARCHAR PRIMARY KEY,
            hmm_domain_name VARCHAR PRIMARY KEY,
            annot VARCHAR,
            evalue FLOAT,
            score FLOAT,
            bias VARCHAR,
            length INTEGER,
            top_hmm_domain_hit_on_prot VARCHAR,
            top_protein_match_for_hmm VARCHAR,
            best_dom_env_start INTEGER,
            best_dom_env_end INTEGER,
            best_dom_env_length INTEGER,
            sequence VARCHAR
        )
        """
        )
    df_for_sql = hmm_scan_df[col_from_hmm_scan]
    query_prot = []
    protein_sequence = []
    for prot in protein_py_easel_obj:
        name = prot.name.decode()
        sequence = prot.textize().sequence
        query_prot.append(name)
        protein_sequence.append(sequence)

    # Create the DataFrame
    data = {"protein_name": query_prot, "sequence": protein_sequence}
    seq_df = pd.DataFrame(data)
    df_for_sql = pd.merge(df_for_sql, seq_df, how="outer", on="protein_name")

    db.insert_dataframe(
        sql_table_name,
        df_for_sql,
        primary_keys=["protein_name", "organism_name", "hmm_domain_name"],
    )


# def get_hmm_list(db_table_name,
#                  db_file_path,
#                  min_bit_score=0):
#     """
#     Get the list of unique/distinct HMM annotations in the table where the min_bit_score is
#     greater than the given value or the HMM is the best hit for a given gene.
#     """
#     this_file_dir = os.path.dirname(os.path.abspath(__file__))
#     logger.info(f'The table to query for hmm hits is {db_table_name}')
#     template_file_path = os.path.join(this_file_dir, 'data_utils/sql_scripts/select_unique_hmms.sql')
#     replacements = {
#         "TABLE_NAME": db_table_name,
#         "MIN_SCORE": f'{min_bit_score}'}
#     query = sql_utils.read_and_replace_query(template_file_path, replacements)
#     hmm_tbl = sql_utils.run_sql_query(db_file_path, query)

#     logger.info(f'The hmm query results are {hmm_tbl}')
#     logger.info(f'hmm result type: {type(hmm_tbl)}')
#     hmm_list = hmm_tbl['hmm_domain_name'].to_list()

#     return hmm_list


# def extract_hmm_annot_sequences_to_file(hmm_annotation: str,
#                                            db_file_path: str,
#                                            db_table_name: str,
#                                            out_folder_path: str,
#                                            min_bit_score=0, # the score is always > 0, so default to no score filter
#                                            out_file_extension: str='fa'):
#     """
#     Get all hits of a particular HMM, eg "phrog_1"
#     This step is required to get the data for an MSA

#     If you want a subset of organisms. The optimal thing to do is to make a new table
#     i.e. an inner join of a table with a organisms lookup and the sequence hmm hits table
#     then use that table in this query (db_table_name)
#     """

#     this_file_dir = os.path.dirname(os.path.abspath(__file__))
#     logger.info(f'The table to query for hmm hits is {db_table_name}')
#     template_file_path = os.path.join(this_file_dir,'data_utils/sql_scripts/select_hmm_sequences.sql')
#     replacements = {
#         "TABLE_NAME": db_table_name,
#         "HMM_ANNOT": hmm_annotation,
#         "MIN_SCORE": f'{min_bit_score}'}
#     query = sql_utils.read_and_replace_query(template_file_path, replacements)
#     hhm_seq_tbl = sql_utils.run_sql_query(db_file_path, query)

#     os.makedirs(out_folder_path, exist_ok=True)
#     out_hmm_fasta = os.path.join(out_folder_path, f'{hmm_annotation}.{out_file_extension}')

#     # write the sql output to a fasta file
#     with open(out_hmm_fasta, 'w') as hmm_fasta_file:
#         for _, row in hhm_seq_tbl.iterrows():
#             query_genome = row['organism_name']
#             query_prot = row['protein_name']
#             sequence = row['sequence']
#             hmm_fasta_file.write(f'>{query_genome};{query_prot}\n')
#             hmm_fasta_file.write(f'{sequence}\n')

# def make_msa(hmm_seq_hits_file: str, #
#              hmm_file_path,
#              out_folder_path: str):
#     '''
#     Given a file of sequences that match an HMM, and the associated HMM file
#     create an MSA file.
#     '''
#     _ , ext = os.path.splitext(out_folder_path)
#     if ext != '.csa':
#         logger.error(f'Expecting an output file with extension ".csa" but got "{ext}"')
#         sys.exit()

#     lst_count_msa_length=[]
#     with pyhmmer.plan7.HMMFile(hmm_file_path) as hmm_file: #Note: Needs to be hmmer3 format - phrogs format does not work
#             hmm = hmm_file.read()
#     hmm_name=hmm.name.decode()
#     in_fasta = hmm_seq_hits_file
#     with pyhmmer.easel.SequenceFile(in_fasta, digital=True) as seqs_file:
#         proteins = seqs_file.read_block()
#         msa = pyhmmer.hmmer.hmmalign(hmm, proteins, trim=False, digitize=False, all_consensus_cols=True)

#         for sequence in msa.alignment:
#             lst_count_msa_length.append(len(sequence))
#             break

#         with open(out_folder_path,'w') as h_hmm_msa:
#             for name,sequence in zip(msa.names, msa.alignment):
#                 h_hmm_msa.write(f'{sequence},{name.decode()}\n') #header: '>'+query_genome+' '+protein_name

# def extract_combined_hmm_summaries(db_file_path: str,
#                                     db_table_name: str,
#                                     out_folder_path: str,
#                                     min_bit_score=0):
#     """
#     This is intended as a replacement for the combine step in the snakemake annotation pipelines
#     It is not an exact replacement because the outputs are dependedant on the values already in the database
#     This is a design decision. SQL operations and MSA logic need to keep the DB content in mind
#     """
#     this_file_dir = os.path.dirname(os.path.abspath(__file__))
#     logger.info(f'The table to query for hmm hits is {db_table_name}')
#     template_file_path = os.path.join(this_file_dir,'data_utils/sql_scripts/select_all_domain_n_genome.sql')
#     replacements = {
#         "TABLE_NAME": db_table_name,
#         "MIN_SCORE": f'{min_bit_score}'}
#     query = sql_utils.read_and_replace_query(template_file_path, replacements)
#     hhm_scan_tbl = sql_utils.run_sql_query(db_file_path, query)
#     counts_table_crosstab = pd.crosstab(hhm_scan_tbl['organism_name'], hhm_scan_tbl['hmm_domain_name'])

#     os.makedirs(out_folder_path, exist_ok=True)
#     out_hmm_fasta = os.path.join(out_folder_path, 'hmm_scan.csv')
#     counts_table_crosstab.to_csv(out_hmm_fasta)
