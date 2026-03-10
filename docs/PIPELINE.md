# Data Pipeline Architecture

This project uses DVC to manage a complex DAG of ~500 lines.
The visualization below is updated whenever the stage definitions change.

![Project DAG](./images/pipeline_dag.png)

## Key Sub-sections
1. **Intake**:
   * The raw sequence data is tracked in DVC at ml_assets/data/sequence_data/
   * The interaction data is extracted from ml assets/data/interaction_data/interactions_vl.duckdb
   * These can both be retrieved from **[Google Drive](https://drive.google.com/drive/folders/1pZeochptPcfzRrc_n76bh0CG-4fjqlvX?usp=drive_link)**
2. **Phage Variants Pipeline**: prodigal_phage->...->aggregate_phage_variants
3. **Host Variants Pipeline**: prodigal_ecoli->...->aggregate_ecoli_variants
4. **Host Serotypes Pipeline**: kaptive_ecoli->aggregate_serotypes
5. **Interaction Data Pipeline**: simple transformation using the extract_ interaction_matrix stage
6. **Training Data Aggregation**:
   * Takes long duckdb tables from 2, 3 and 4 above
   * And integrates it with the interaction matrix from 5
   * Produces HDF5 files for the AlphaPhage Transformer
7. **Training**: Trains AlphaPhage Transformer.
