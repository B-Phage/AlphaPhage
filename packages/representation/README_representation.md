## representation

The representation package is responsible for turning domain objects into model-ready inputs.

It contains logic for encoding biological data, applying vocabularies, transforming alignments, and assembling training data in the format expected by downstream models.

This package defines how data is represented for learning, independent of how it is stored, orchestrated, or biologically grouped.

In short:

* variants decides what the biological units are

* representation decides how they are encoded for the model

The internal structure may evolve, but the responsibility remains the same:
to provide a clear, consistent bridge between domain data and model input.
