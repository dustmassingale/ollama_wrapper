# Contributing to Ollama Wrapper RAG Agent

Thank you for your interest in contributing to this project! Here are some guidelines to help you get started.

## Prerequisites

- A modern version of Node.js or Python (depending on your preferred environment)
- Knowledge of the relevant programming languages used in the project
- A basic understanding of software development best practices
- Familiarity with Git and version control
- Access to Ollama for local testing (if applicable)

## Getting Started

1. **Fork the repository**: Use GitHub's fork functionality to create your own copy of the project

2. **Clone your fork locally**:
   ```bash
   git clone https://github.com/your-username/ollama_wrapper.git
   cd ollama_wrapper/rag_agent
   ```

3. **Install dependencies**: Use `npm` for Node.js packages or `pip` for Python packages

4. **Configure environment**: Copy `.env.example` to `.env` and configure your settings as needed

5. **Build the project**: Follow build instructions from `README.md`

## Code Style

- We follow standard ESLint/Prettier conventions for JavaScript/TypeScript projects
- For Python projects, we use Black and flake8 for consistent formatting
- Indentation: 2 spaces for JavaScript/TypeScript, 4 spaces for Python
- Add descriptive inline comments when necessary
- Keep functions and classes self-contained and well-documented

## Pull Request Process

1. **Create a new branch** for your feature or fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** and commit them using clear, descriptive messages following conventional commit format

3. **Test your changes locally**: Ensure all tests pass before submitting

4. **Push your branch** to GitHub

5. **Open a pull request** with:
   - A clear title starting with the issue number (e.g., `feat: implement document indexing`)
   - A detailed description of what you changed and why
   - References to related issues or discussions

## Issues and Bug Reports

- Check existing issues before creating a new one
- Clearly describe the problem or bug you're reporting
- Include steps to reproduce the issue if applicable
- Provide expected vs actual behavior
- Attach logs, screenshots, or other diagnostic information when relevant

## Security Concerns

- Report any security vulnerabilities privately via our private channels
- Do not disclose sensitive information in public forums
- Follow responsible disclosure practices

## Communication

We value clear and respectful communication. All contributors should be prepared to discuss their work openly and collaboratively. Feel free to reach out if you have questions about the contribution process or need guidance with your implementation.