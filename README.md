# Notebooks Template

> [!IMPORTANT]
> Adapt the headline and description text to your needs.

This repo is a template for repositories that contain merely jupyter notebooks.
Please create a new repository for each topic using this template to keep things small and simple (avoiding dependency conflicts).

## Usage

We use [`uv`](https://docs.astral.sh/uv) to manage the dependencies (because it is faster and simpler than `pipenv` or `hatch`).

> [!IMPORTANT]
> Adapt the metadata in `pyproject.toml` to your needs.

To create the local Python environment use

```sh
uv sync
```

Then to run JupyterLab within the environement use

```sh
uv run jupyter lab
```

To add a dependency use (the `"` are important)

```sh
uv add "package-name~=min-version"
```

## Contents

> [!IMPORTANT]
> List and explain here the purpose of each notebook.

### `Example.ipynb`

This notebook contains just a stupid example.

## License

This repository is licensed under the terms of the [MIT License](LICENSE).

## Developer Notice

If you want to commit changes to this repository or ones based on this template, do not forget to activate `pre-commit` for correct formatting and notebook output stripping in advance via

```sh
uv run pre-commit install
```

The formatting and stripping is peformed on all files added to the git staging area (via `git add`).
You may run it on all files in the repository using

```sh
uv run pre-commit run --all
```
