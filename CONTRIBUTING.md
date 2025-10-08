# Contribution Guidelines for unity-ml-drl-data

This document outlines the guidelines for making contributions to ensure consistency and maintain code quality. Please review it carefully before submitting your work.
&nbsp;

## Prerequisites
All development must be done using the following software versions to ensure project compatibility:
* **Unity Editor:** `2023.2.12f1`
* **Python:** `3.10.12` (preferred) or `3.10.11`.

## Contribution Workflow
This project utilizes the Fork & Pull Request workflow for all code contributions.

### 1. Fork the Repository
First, create a personal fork of the main repository by clicking the "Fork" button on the project's primary GitHub page.

### 2. Clone Your Fork
Clone your forked repository to your local development environment.

```bash
git clone https://github.com/omarelfiki/unity-ml-drl-data.git
cd unity-ml-drl-data
```

### 3. Create a New Branch
All changes must be made in a feature branch, not directly on the main branch. This isolates work and facilitates the review process.

Branch names should be descriptive. Use the following conventions:

* **For new features:** `feature/feature-description`

* **For bug fixes:** `fix/bug-description`

Example:

```Bash
git checkout -b feature/implement-ai-heuristic
```

### 4. Commit Changes
Implement your changes and commit them with a clear, descriptive message. It is highly recommended to follow the Conventional Commits specification for commit messages.

Example:
```bash
git add .
git commit -m "feature: Implements new ai heuristic that lowers run time"
```

### 5. Push to Your Fork
Push your new branch and its commits to your remote fork on GitHub.

Example:
```bash
git push origin feature/implement-ai-heuristic
```

### 6. Open a Pull Request (PR)
Navigate to your forked repository on GitHub. Initiate a pull request from your feature branch to the main branch of the original repository.

## Pull Request Guidelines
All pull requests must meet the following criteria to be considered for merging:
* **A clear and descriptive title:** Summarize the purpose of the changes concisely.
* **A detailed summary:** Explain the changes made and the reasoning behind them.
* **Link to a relevant issue:** If the pull request resolves an existing issue, reference it in the description (e.g., Closes #42).

## Code Standards
All contributions must adhere to the following code standards to maintain project quality and consistency:

* **Meaningful Naming Conventions:** Use clear and descriptive names for all variables, functions, and classes.
* **Consistent Code Formatting:** Adhere to the existing code style and formatting present in the project.
* **Comprehensive Documentation:** Add comments to clarify complex logic or non-obvious code segments.
* **Effective Error Handling:** Implement robust error handling where applicable.
* **DRY (Don't Repeat Yourself) Principle:** Avoid code duplication by creating reusable functions or classes.

## gitignore 
A ".gitignore" file tells Git which files/folders it should ignore - they won't get committed. 
This helps with : 
* Keeping the repository clean
* Avoids committing unnecessary files
* Prevents merge conflicts


## Training Runs Naming convention 
Every group member must follow this naming convention when they start a training run.

X_Y_E(_A)

* X = Your initials (eg. LF)
* Y = the date you make the training run (eg. 011025 (01/10/2025))
* E = the name of the enviroment you are training on (eg. Basic)
* A = Optional (if running the same enviroment multiple times in one day)
Example of file name: **LF_100125_Basic_1**

The file on tensorboard will look like the following, LF_100125_Basic_1/Basic (/Basic will be filled out automatically dependning on the training scene you choose)

