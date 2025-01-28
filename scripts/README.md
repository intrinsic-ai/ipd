# Scripts to download and extract the dataset, cad models

## Option 1: IPDReader

See `ipd/demo_reader.ipynb`. The `IPDReader` class will download and read the dataset.

## Option 2: Bash Script

Will download one specified dataset, and extract it. Then will download all cad models to `models` subfolder.

```bash
export ID=dataset_basket_0
export CAMERA=Basler-LR
export FOLDER=./datasets

bash scripts/get_dataset.sh $ID $CAMERA $FOLDER
```

## Option 3: Python CLI

Has options to download one or more datasets to specified folder, option to extract (will not extract by default). Then will download all cad models to `models` subfolder.

- Install the `intrinsic-ipd-cli`:
    - From source:
        1. Clone this repo
        2. Install `pip install -e .`
    - Via pip: (not yet available!!!)
        1. `pip install ipd`
    
    Should have the download cli available via `ipd-cli` command.

- To download and extract all datasets:
    ```bash
    intrinsic-ipd-cli --id ALL --camera ALL --root ./datasets --extract
    ```

- To download and extract one dataset:
    ```bash
    intrinsic-ipd-cli --id dataset_basket_1  --camera Basler-LR --root ./datasets --extract
    ```

- To download and extract all cameras for one dataset:
    ```bash
    intrinsic-ipd-cli --id dataset_basket_1  --camera ALL --root ./datasets --extract
    ```

- To download and extract all datasets for one camera:
    ```bash
    intrinsic-ipd-cli --id ALL  --camera Basler-LR --root ./datasets --extract
    ```

- All command line options:
    ```bash
    intrinsic-ipd-cli -h

    options:
    -h, --help              show this help message and exit
    --camera                    {ALL, FLIR_polar, Photoneo, Basler-HR, Basler-LR}
                            Supply a camera id or or 'ALL' to download all cameras for the specified dataset
    --id                        {ALL, dataset_basket_0, dataset_basket_1,..., dataset_texturedbg_3}
                            Supply a dataset id or 'ALL' to download all datasets for specified camera
    --root FOLDER           Folder to download/extract the datasets to
    --extract               Flag to extract the downloaded dataset(s)
    ```
