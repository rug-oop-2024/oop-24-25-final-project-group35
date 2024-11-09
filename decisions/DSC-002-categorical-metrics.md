# DSC-002: Handling Categorical Labels in Metrics

- **Date**: 2023-10-31
- **Decision**: Change string labels to integers in metric functions to handle categorical labels.
- **Status**: Accepted
- **Motivation**: To fix errors occurring when metrics like Cohen's Kappa are applied to datasets with string labels.
- **Reason**: Ensures that metrics can be calculated correctly for classification tasks with string labels.
- **Limitations**: Additional processing overhead, need to ensure that the mapping is consistent and accurate.
- **Alternatives**: Require that all labels be converted to integers before metric computation or use external libraries that handle string labels.

