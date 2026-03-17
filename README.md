## Installation

This project uses [**uv**](https://docs.astral.sh/uv/) for environment and dependency management.  

On Linux or macOS:
```sh
uv venv
source .venv/bin/activate
```
On Windows: 
```sh
uv venv
.venv\Scripts\activate
```

### Default installation
Move into the project directory and run 
```sh
uv pip install .
```

### Developer installation
Install the project in editable mode:
```sh
uv pip install -e ".[dev]"
```
and install the pre-commit hook
```sh
pre-commit install
```
