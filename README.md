# Treasure tokens (treakens)
Welcome to the worst named repo ever.

## Setup

1. [Install Pixi](https://pixi.sh/latest/installation/).
2. Clone this repository and navigate to the top level of the repository.
3. Run

```
pixi shell
```

This will launch a shell and install all necessary dependencies.
Inside of this environment the code under `src/treakens` will behave like a library.
So we can do imports like `from treakens.model import kmeans` for example.

TODO: check that torch CUDA actually gets installed on s3df!

## Resources

* [Pixi documentation](https://pixi.sh/)
* [Reproducible Machine Learning Workflows for Scientists, Matthew Feickert, 2025](https://carpentries-incubator.github.io/reproducible-ml-workflows/)
* [Good ML project structure](https://github.com/mattcleigh/JetSSL-Lite/tree/master)