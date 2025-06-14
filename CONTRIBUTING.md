# Contributing to FAL.ai MCP Server

Thanks for your interest in contributing! This guide will help you get started with development and ensure your contributions align with the project standards.

## 🚀 Quick Start

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/yourusername/fal-mcp.git
   cd fal-mcp
   ```

2. **Set up development environment**
   ```bash
   make setup
   # This installs dependencies and sets up pre-commit hooks
   ```

3. **Configure your API key**
   ```bash
   cp .env.example .env
   # Edit .env with your FAL_KEY
   ```

4. **Verify everything works**
   ```bash
   make validate
   ```

## 🛠️ Development Workflow

### Before You Start
- Check existing issues and PRs to avoid duplicate work
- Create an issue for significant changes to discuss the approach
- Fork the repository and create a feature branch

### Making Changes

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Follow the coding standards (see below)
   - Add tests for new functionality
   - Update documentation as needed

3. **Run quality checks**
   ```bash
   make run-checks  # Runs all checks
   # Or individually:
   make format      # Auto-format code
   make lint        # Check code style
   make type-check  # Run type checking
   make test        # Run tests
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add new feature description"
   ```
   Pre-commit hooks will automatically run and may modify files.

5. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

## 📋 Code Standards

### Python Style
- **Line length**: 88 characters (Black default)
- **Import sorting**: Use isort with Black profile
- **Type hints**: Add type hints for public functions
- **Docstrings**: Use Google-style docstrings for public functions

### Code Quality Tools
We use several tools to maintain code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking
- **bandit**: Security scanning
- **pydocstyle**: Docstring style checking

All tools are configured in `pyproject.toml` and run automatically via pre-commit hooks.

### Commit Messages
Follow conventional commit format:
- `feat:` - New features
- `fix:` - Bug fixes
- `docs:` - Documentation changes
- `style:` - Code style changes (formatting, etc.)
- `refactor:` - Code refactoring
- `test:` - Adding or updating tests
- `chore:` - Maintenance tasks

Examples:
```
feat: add video generation with Runway Gen3
fix: handle API timeout errors gracefully
docs: update installation instructions
```

## 🧪 Testing

### Running Tests
```bash
make test           # Run all tests
make test-verbose   # Run with verbose output
```

### Writing Tests
- Add tests for new functionality in `test_server.py`
- Use descriptive test names: `test_generate_image_with_valid_params`
- Mock external API calls to avoid hitting real endpoints
- Test both success and error cases

### Test Structure
```python
def test_your_feature():
    """Test description of what this validates."""
    # Arrange
    client = FALClient("test-key")

    # Act
    result = client.your_method()

    # Assert
    assert result is not None
    assert "expected_field" in result
```

## 🔒 Security

### API Keys
- Never commit real API keys
- Use `.env` files for local development
- Test with mock/fake keys when possible

### Dependencies
- Keep dependencies up to date
- Run `make security-check` to scan for vulnerabilities
- Review new dependencies for security issues

### Code Security
- Validate all user inputs
- Use parameterized queries/requests
- Avoid eval() or exec() calls
- Handle errors gracefully without exposing internals

## 📚 Documentation

### Code Documentation
- Add docstrings to all public functions and classes
- Include parameter types and return types
- Provide usage examples for complex functions

### README Updates
- Update README.md if you add new tools or change functionality
- Keep installation instructions current
- Update examples if API changes

### API Documentation
- Document new MCP tools in the README
- Include parameter schemas and examples
- Explain any breaking changes

## 🐛 Bug Reports

When reporting bugs, include:
- Python version and OS
- FAL.ai MCP server version
- Steps to reproduce
- Expected vs actual behavior
- Error messages and stack traces
- Minimal code example if possible

## 💡 Feature Requests

For new features:
- Check if it aligns with the project goals
- Provide use cases and examples
- Consider backward compatibility
- Discuss implementation approach in an issue first

## 🔄 Release Process

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Run full test suite
4. Create release PR
5. Tag release after merge
6. Update documentation

## 📞 Getting Help

- **Issues**: For bugs and feature requests
- **Discussions**: For questions and general discussion
- **Email**: For security issues (don't create public issues)

## 🎯 Development Tips

### Useful Make Commands
```bash
make help           # Show all available commands
make quick-check    # Fast checks during development
make clean          # Clean up generated files
make check-env      # Verify environment setup
make dev-server     # Run server for testing
```

### IDE Setup
For VS Code, install these extensions:
- Python
- Black Formatter
- isort
- Pylance
- GitLens

### Debugging
- Use `python -m pdb main.py` for debugging
- Add `breakpoint()` calls in your code
- Check logs with verbose output

## 🤝 Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Follow the project's technical standards

---

**Questions?** Feel free to open an issue or start a discussion. We're here to help!
