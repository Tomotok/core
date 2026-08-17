## Development setup

Run commands from the core repository root (`core/`).

```bash
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install -r requirements.txt
```

Use Python 3.10 or newer.

## Testing

Run the full test suite before opening a merge request.

```bash
python -m unittest discover -s tomotok -p "test_*.py"
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

## Commit message rules

- Start with a verb in imperative form
- Be reasonably specific
- Example: `Fix documentation for weighted squares regularisation matrix`

## Docstrings (NumPy style)

Use NumPy-style docstrings as described here:
https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html

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

## Merge request checklist

Before requesting review:
- Run tests locally
- Update or add tests for behavior changes
- Update docs for public API changes
- Ensure commit messages follow the rules above
