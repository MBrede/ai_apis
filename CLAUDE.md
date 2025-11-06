## Coding Guidelines & Best Practices

### Code Quality Standards
- **Type Hints**: Use type annotations for all function signatures and class attributes
- **Docstrings**: Follow Google/NumPy docstring format for all modules, classes, and functions
- **Error Handling**: Use specific exception types with descriptive messages
- **Logging**: Use proper logging instead of print statements for debugging
- **Testing**: Write unit tests using pytest for all core functionality

### Python Best Practices
- **PEP 8**: Follow Python style guide strictly (automated with black + isort)
- **Single Responsibility**: Each class/function should have one clear purpose
- **DRY Principle**: Don't repeat yourself - extract common code into utilities
- **SOLID Principles**: Especially Interface Segregation (our interface-based design)
- **Context Managers**: Use `with` statements for file operations and resource management

### Package Structure & Imports
- **Absolute Imports**: Always use `from FaceSwap.src.module import Class`
- **Interface Inheritance**: All implementations must inherit from base interfaces
- **Package Namespace**: Respect the `FaceSwap.src.*` namespace structure
- **Lazy Loading**: Import heavy dependencies only when needed
- **Circular Imports**: Avoid by using forward references and runtime imports
- **Up-to-date versions**: Do look up the most recent compatible versions of modules, do not trust your prior knowledge.

### Configuration Management
- **Config Singleton**: Use `Config()` singleton for all configuration access
- **YAML Over JSON**: Prefer YAML for human-readable configuration files
- **Environment Variables**: Support environment variable overrides for deployment
- **Path Management**: Use `Constants` class for all project paths
- **Pathlib Usage**: Always use `pathlib.Path` for cross-platform compatibility (Windows/Linux)
- **Validation**: Validate config values at startup with descriptive error messages

### Error Handling & Debugging
- **Specific Exceptions**: Use `FileNotFoundError`, `ValueError`, etc. over generic `Exception`
- **Error Context**: Include relevant context (file paths, parameter values) in error messages
- **Graceful Degradation**: Handle missing optional dependencies gracefully
- **Stack Traces**: Preserve full traceback information for debugging
- **Early Validation**: Validate inputs at function entry points

### Performance & Memory
- **Memory Efficiency**: Use generators for large datasets, avoid loading everything into memory
- **Caching**: Cache expensive operations (model loading, feature extraction)
- **Batch Processing**: Process data in batches rather than one-by-one
- **Resource Cleanup**: Properly close files, models, and GPU resources
- **Progress Tracking**: Use tqdm for long-running operations

### Machine Learning Specific
- **Device Agnostic**: Write code that works on CPU/GPU without hardcoding
- **Model Interfaces**: Use our defined interfaces for all ML components
- **Checkpoint Management**: Save/load models consistently with metadata
- **Reproducibility**: Set random seeds for deterministic behavior
- **Data Validation**: Validate tensor shapes and data types early

### Git & Version Control
- **Atomic Commits**: Each commit should represent one logical change
- **Descriptive Messages**: Write clear commit messages explaining the "why"
- **Branch Naming**: Use descriptive branch names (feature/face-clustering)
- **No Large Files**: Use Git LFS for models, datasets, and binaries
- **Clean History**: Squash commits before merging to maintain clean history

### Testing Strategy
- **Unit Tests**: Test individual functions/classes in isolation
- **Integration Tests**: Test component interactions (data pipeline, model training)
- **Smoke Tests**: Quick tests to verify basic functionality works
- **Property-Based Testing**: Use hypothesis for testing edge cases
- **Mock External Dependencies**: Mock file systems, network calls, GPU operations

### Documentation Requirements
- **API Documentation**: Document all public interfaces and their contracts
- **Usage Examples**: Provide working examples for each major component
- **Configuration Guide**: Document all config options and their effects
- **Architecture Decisions**: Document why certain design choices were made
- **Troubleshooting**: Common issues and their solutions

### Code Review Checklist
1. ✅ Type hints on all function signatures
2. ✅ Proper exception handling with specific types
3. ✅ Docstrings following Google format
4. ✅ No hardcoded paths or magic numbers
5. ✅ All paths use `pathlib.Path` for cross-platform compatibility
6. ✅ Proper resource cleanup (files, models, memory)
7. ✅ Unit tests for new functionality
8. ✅ Configuration managed through Config singleton
9. ✅ Imports follow package namespace conventions
10. ✅ Error messages are descriptive and actionable
11. ✅ Code follows single responsibility principle
