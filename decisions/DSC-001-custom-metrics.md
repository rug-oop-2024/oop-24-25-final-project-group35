# DSC-001: Implement Custom Metrics Without External Libraries

- **Date**: 2023-10-23
- **Decision**: Implement custom metrics directly without importing external libraries.
- **Status**: Accepted
- **Motivation**: This was made to ensure that the metrics like the mean squared error, accuracy, Cohen's kappa, mean absolute error, R^2 and CSI are calculated directly and to avoid dependencies on external libraries, which is required for the project.
- **Reason**: Shows a deeper understanding of metric calculations and adheres to the requirement of direct implementations.
- **Limitations**: Required more time and effort to implement and test custom metrics, potentiallly errors if not carefully implemented.
- **Alternatives**: Use existing implementations from libraries like scikit-learn or NumPy.
