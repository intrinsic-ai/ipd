# Scripts to download and extract the dataset, cad models

## Option 1: Bash Script

Will download one specified dataset, and extract it. Then will download all cad models to `models` subfolder.

```bash
export ID=dataset_basket_0
export CAMERA=Basler-LR
export FOLDER=./datasets

bash scripts/dataset/get_dataset.sh $ID $CAMERA $FOLDER
```

## Option 2: Python CLI

Has options to download one or more datasets to specified folder, option to extract (will not extract by default). Then will download all cad models to `models` subfolder.

- Install the `ipd-cli`:
    - From source:
        1. Clone this repo
        2. Install `pip -e .[cli]`
    - Via pip:
        1. `pip install ipd[cli]`
    
    Should have the download cli available via `ipd-cli` command.

- To download and extract all datasets:
    ```bash
    ipd-cli --id ALL --camera ALL --folder ./datasets --extract
    ```

- To download and extract one dataset:
    ```bash
    ipd-cli --id dataset_basket_1  --camera Basler-LR --folder ./datasets --extract
    ```

- To download and extract all cameras for one dataset:
    ```bash
    ipd-cli --id dataset_basket_1  --camera ALL --folder ./datasets --extract
    ```

- To download and extract all datasets for one camera:
    ```bash
    ipd-cli --id ALL  --camera Basler-LR --folder ./datasets --extract
    ```

- All command line options:
    ```bash
    ipd-cli -h

    options:
    -h, --help              show this help message and exit
    --camera                    {ALL, FLIR_polar, Photoneo, Basler-HR, Basler-LR}
                            Supply a camera id or or 'ALL' to download all cameras for the specified dataset
    --id                        {ALL, dataset_basket_0, dataset_basket_1,..., dataset_texturedbg_3}
                            Supply a dataset id or 'ALL' to download all datasets for specified camera
    --folder FOLDER         Folder to download/extract the datasets to
    --extract               Flag to extract the downloaded dataset(s)
    ```