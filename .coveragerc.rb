# .coveragerc
[run]
# Specify the location of your source code (useful if not using --cov flag)
source = utils
# Exclude files that you never want to measure
omit =
    # Example: Exclude database migrations
    */migrations/*
    # Example: Exclude settings files
    */settings.py
    # Example: Exclude virtual environment paths
    .venv/*
    # Exclude files where all code is boilerplate, like __init__.py files
    __init__.py

[report]
# Require a minimum coverage percentage for the entire project
fail_under = 80 # TODO: up this

# Show lines that were skipped because they were marked as 'pragma: no cover'
show_missing = True

[html]
# Directory where the HTML report will be generated
directory = htmlcov
