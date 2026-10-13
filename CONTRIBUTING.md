## Development setup

Run commands from the core repository root (`core/`).

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[docs]"
```

Use Python 3.10 or newer.

## Testing

Run the full test suite before opening a pull/merge request.

```bash
./dev/run_tests.sh
```

Equivalent to:

```bash
python -m unittest discover -s tests -t . -p "test_*.py"
```

Recommended for bug fixes and features:
- Add or update tests for changed behavior
- Keep optional-backend tests guarded when dependencies are not installed

## Documentation build

Regenerate API pages and HTML docs when public API or doc pages change.

```bash
./dev/apidoc.sh
./dev/docs_html.sh
```

Figures in the documentation are not generated during the build. They are created by scripts stored next to
them (e.g. `docs/source/methods/figures/make_bob_figures.py`). When a change of the code affects a figure,
rerun the corresponding script and commit the updated figures. `tests/test_docs_figures.py` runs the scripts
on a small configuration to check that they work with the current code.

## Copyright and licence notices

- New source files start with the licence header using the current year:

  ```python
  # Copyright 2026 Institute of Plasma Physics of the Czech Academy of Sciences.
  # Licensed under the EUPL-1.2 or later.
  ```

  Headers of existing files keep the year of their creation.
- In the first change of each new year, update the end year of the copyright notice in `docs/source/conf.py`
  (e.g. `2021–2026` to `2021–2027`).
- When a new author or contributor joins, add them to `AUTHORS.txt`, to the contributors tables in `README.md`
  and `docs/source/welcome.rst`, to the `authors` of `CITATION.cff` and to `author` in `docs/source/conf.py`.

## Commit message rules

- Start with a verb in imperative form
- Be reasonably specific
- Example: `Fix documentation for weighted squares regularisation matrix`

## Docstrings (NumPy style)

Use NumPy-style docstrings as described here:
https://numpydoc.readthedocs.io/en/latest/format.html

Project conventions compatible with NumPy style:
- For multiline docstrings, place the opening triple quotes on their own line and start the summary on the next line
- Keep the summary short and end it with a period
- Leave one blank line between summary and the next section
- Include standard sections as relevant: `Parameters`, `Returns`, `Raises`, `Notes`, `Examples`

For classes:
- Document class behavior and attributes in the class docstring
- Document constructor parameters in `__init__` when it improves clarity and avoids ambiguity
- Do not duplicate the same long parameter docs in both class and `__init__`

Example:

```python
class Foo:
    """
    Class summary goes here.

    More detailed description of class behavior can be placed here. 
    It can span multiple lines and paragraphs as needed.

    Attributes
    ----------
    attr : float
        Sum of the two constructor parameters.
    """

    def __init__(self, param1, param2):
        """
        Summary of init method should be placed here.

        Parameters
        ----------
        param1 : float
            First value.
        param2 : float
            Second value.
        """
        self.attr = param1 + param2
```

## Pull/merge request checklist

Before requesting review:
- Run tests locally
- Update or add tests for behavior changes
- Update docs for public API changes
- Ensure commit messages follow the rules above
