# Bug Report

---
name: Bug Report
description: Report a bug or issue
body:
- type: textarea
  attributes:
    label: Describe the bug
    description: A clear and concise description of what the bug is.
  validations:
    required: true
- type: textarea
  attributes:
    label: To Reproduce
    description: Steps to reproduce the behavior.
    placeholder: |
      1. Go to '...'
      2. Click on '...'
      3. See error
  validations:
    required: true
- type: textarea
  attributes:
    label: Expected behavior
    description: A clear and concise description of what you expected to happen.
- type: textarea
  attributes:
    label: Screenshots
    description: If applicable, add screenshots to help explain your problem.
- type: input
  attributes:
    label: Environment
    description: Please provide details about your environment.
    placeholder: OS, Python version, etc.
- type: textarea
  attributes:
    label: Additional context
    description: Add any other context about the problem here.
