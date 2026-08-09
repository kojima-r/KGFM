"""Enable `python -m kgfm ...` (and `torchrun ... -m kgfm ...`)."""

from .cli import main

if __name__ == "__main__":
    main()
