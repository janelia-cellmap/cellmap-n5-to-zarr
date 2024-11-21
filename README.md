This script takes paths to raw/label n5 datasets as an input, and outputs .zarr container with the correct cellmap schema:

  n5 / zarr container
  - recon_{number}
    - em
        * fibsem-uint8
        * fibsem-uint16
    - labels
      * inference (predictions and segmentations)
      * groundtruth (training crops)
      * mask
      
Installation:
  1. cd PATH_TO_POETRY_PROJECT_DIRECTORY/
  2. poetry install

Example(for lsf cluster):

  bsub -n 15 -J n5convert -o path_to_output_log_file 'poetry run python src/to_cellmap_zarr.py --num_workers=300 --cluster=lsf --src=path_to_raw_data_n5_group(array) --inf=path_to_inference --dest=path_to_the_output_zarr_container';
