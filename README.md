To create and activate the environment:

```
conda env create -f environment.yml
conda activate im2txt
```

If adding any new requirements, add them to requirements.in and do:

`pip-compile requirements.in`

To install the requirements in already generated requirements.txt, do:

`pip-sync requirements.txt`

To add the environment to the jupyter notebook, do:

`python -m ipykernel install --user --name=im2txt`
