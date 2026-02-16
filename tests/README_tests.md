## 1️⃣ Unit tests vs Integration tests
|Feature | Unit tests | Integration tests
| :--- | :--- | :--- |
|Scope | Single functions, classes, or modules in isolation | Multiple modules / packages working together, or “full workflows” |
| Dependencies | Mock external calls, binaries, databases, APIs | Use real components where feasible, but may mock heavy dependencies |
|Goal | Verify logic correctness for each building block | Verify end-to-end behavior of a workflow/pipeline |
|Coverage target | Ideally 100% for package code | Focus on critical paths, not necessarily 100% of workflow code |

## 2️⃣ High-level goals for integration tests in a production-grade bioinformatics repo
1. End-to-end correctness
   * Run workflows on representative datasets.
   * Verify outputs exist and have expected structure/content.
   * Ensure DVC stages or orchestrator stages run as expected.
2. Environment sanity
   * Confirm that conda environments, binaries, and external tools (like prodigal-gv) are properly invoked.
   * Detect missing or misconfigured dependencies.
3. Config and parameter handling
   * Validate that workflows correctly consume config files, params.yaml, or command-line options.
   * Check that changes in configs trigger correct behavior (e.g., re-run or skip stages).
4. Parallelism / concurrency
   * Verify that orchestrator or parallel execution works without deadlocks or race conditions.
   * Ensure outputs are correctly written when running in multiple processes.
5. Error handling and robustness
   * Confirm that failures are caught and logged.
   * Ensure partial runs do not leave inconsistent output.
   * Test that the workflow can resume from intermediate states if appropriate.
6. Data integrity
   * Validate that outputs are consistent with inputs:
        * Correct file types
        * Correct number of records
        * Checksums / hash comparisons where appropriate
7. Reproducibility
   * Confirm that repeated runs produce identical outputs if inputs and configs are unchanged.
   * Optional: verify hash tracking for caching/skip logic (like your Orchestrator or DVC stages).
8. Minimal necessary coverage
   * Unlike unit tests, you rarely aim for 100% line coverage in integration tests.
   * Focus on critical workflows, edge cases, and interaction between modules.

## 3️⃣ Practical test philosophy
* Unit tests = 100% coverage of package code
    * Small, fast, deterministic, mock external dependencies
* Integration tests = high confidence in workflow correctness
    * Test key pipelines, not every line
    * Use representative but small datasets to keep runtime reasonable
    * Test config variations, parallel runs, external tool integration

**Rule of thumb:**
Integration tests should focus on _“does the workflow deliver correct results with all dependencies in place?”_ rather than _“does every line of orchestrator code execute?”_

## 4️⃣ Examples of good integration test scenarios for our repo
1. Small genome → prodigal → proteins
   * Ensure proteins exist, correct number of sequences, hashes updated
2. Multiple genomes with different config files
    * Test per-dataset config handling, multi-output handling, parallel execution
3. Protein → HMM annotation workflow
   * Test that HMM outputs are generated correctly and match expectations
4. Force rerun vs cached run
    * Confirm that rerun happens when code or config changes
    * Confirm that skipped when nothing changed
5. Error simulation
   * Provide a broken genome file and confirm orchestrator logs error but continues with other genomes
