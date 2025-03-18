The document is under construction.
# Data-Mining-Project
TCS 2021's Class submission for the course Data Mining - 20XT83

### We intend to use these versions throughout the project
* Python: 3.11.x 
* NumPy: Version 2.2.0
* Pandas: Version 2.0.0
* Scikit-learn: Version 1.6.0
* Streamlit : Version 1.41.0

## Installation

### Dependencies
pytest ~= 6.2.4
pylint ~= 2.9.6

## GitHub Actions: Code Quality Checks

Our project uses GitHub Actions to automatically run code quality checks on all pull requests. Please ensure your contributions pass these checks before submitting.

### PyLint Code Quality Requirements

All code must pass the following PyLint checks([Reference](https://github.com/sprytnyk/pylint-errors?tab=readme-ov-file)):

#### Basic Checker Requirements
| Code | Description | 
|------|-------------|
| C0103 | Use valid and descriptive variable/function names |
| C0112 | Don't use empty docstrings |
| C0114 | Include module docstrings |
| C0116 | Include function docstrings |
| E0102 | Avoid function redefinition |
| R5501 | Use `elif` instead of `else` followed by `if` |
| C0301 | Keep lines within reasonable length |
| E0108 | Don't use duplicate argument names |

#### Import Style Requirements
| Code | Description |
|------|-------------|
| C0410 | Avoid multiple imports on the same line |
| W0611 | Remove unused imports |
| C0411 | Follow proper import order |
| C0415 | Place imports at the top of the file |

#### Code Duplication Check
| Code | Description |
|------|-------------|
| R0801 | Avoid duplicate code |

#### Refactoring Requirements
| Code | Description |
|------|-------------|
| R1714 | Use `in` operator when appropriate |
| W0612 | Remove unused variables |
| W0613 | Remove unused function arguments |
| W0614 | Avoid unused wildcard imports |

### PyTest Requirements

Try achieving a coverage score >80. 

These checks help maintain code quality and consistency throughout the project. Pull requests failing these checks will need revision before merging.

Please use the valid channel on [discussions tab](https://github.com/TCS-2021/Data-Mining-Project/discussions) for any other queries.