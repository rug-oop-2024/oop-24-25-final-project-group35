# DSC-003: Using 'st.session_state' for State Management in Streamlit

- **Date**: 2023-11-04
- **Decision**: Use 'st.session_state' to manage user interactions and state persistence in the Streamlit app.
- **Status**: Accepted
- **Motivation**: To ensure that user interactions, such as deletion confirmations, persist across script reruns and provide a smooth user experience.
- **Reason**: Streamlit re-runs the script on each user interaction; using `st.session_state` allows for state persistence and proper flow control.
- **Limitations**: Requires careful management to prevent state-related bugs, which may add complexity to the code.
- **Alternatives**: Use other state management techniques, such as hidden form fields or URL parameters.


