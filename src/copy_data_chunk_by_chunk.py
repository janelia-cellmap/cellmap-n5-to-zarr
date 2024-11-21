import os
import zarr
import dask.array as da

from dask.distributed import Client
import time 
import numpy as np
from numcodecs import Zstd
from typing import Literal, Tuple, Union
from dask.array.core import slices_from_chunks, normalize_chunks
from toolz import partition_all
from dask.distributed import wait


def save_chunk(
        source: zarr.Array, 
        dest: zarr.Array, 
        out_slices: Tuple[slice, ...],
        invert: bool):
    
    in_slices = tuple(out_slice for out_slice in out_slices)
    source_data = source[in_slices]
    # only store source_data if it is not all 0s
    if not (source_data == 0).all():
        if invert == True:
            dest[out_slices] = np.invert(source_data)
        else:
            dest[out_slices] = source_data
    return 1

def copy_arrays_data(src_dest_info,
                     zs,
                     client: Client,
                     num_workers: int,
                     invert: bool,
                     comp):
    for src_obj, dest_group in src_dest_info:

        # if isinstance(src_obj, zarr.core.Array):
        #     zarrays = [(src_obj.basename, src_obj)]
        # else:
        #     zarrays = src_obj.arrays(recurse = True)
        client.cluster.scale(num_workers)

        if isinstance(src_obj, zarr.core.Array):
            zarrays = [src_obj]
        else:
            zarrays = [key_val_arr[1] for key_val_arr in (src_obj.arrays(recurse=True))]
        
        for arr_src in zarrays:
            start_time = time.time()
            #arr_src = item[1]

            #the chunk sizing of a dask array has a very big impact on computation performance
            # for example: for ~10TB dataset, use ~300MB chunk size.
            #darray = da.from_array(arr_src, chunks=optimal_dask_chunksize(arr_src, max_dask_chunk_num))
            
            if isinstance(src_obj, zarr.core.Array):
                dest_arr = zarr.open(store = zs,
                                    path = dest_group,
                                    mode='w',
                                    shape=arr_src.shape,
                                    chunks=arr_src.chunks, 
                                    dtype=arr_src.dtype,
                                    compressor=comp)
            else:
                arr_path = arr_src.path.replace(src_obj.path, '')
                dest_arr = zarr.open(store =zs,
                                    path=os.path.join(dest_group.lstrip("/"), arr_path.lstrip("/")),
                                    mode='w',
                                    shape=arr_src.shape,
                                    chunks=arr_src.chunks,
                                    dtype=arr_src.dtype,
                                    compressor=comp)
            
            out_slices = slices_from_chunks(normalize_chunks(dest_arr.chunks, shape=dest_arr.shape))
            # break the slices up into batches, to make things easier for the dask scheduler
            out_slices_partitioned = tuple(partition_all(100000, out_slices))
            
            for idx, part in enumerate(out_slices_partitioned):
                print(f'{idx + 1} / {len(out_slices_partitioned)}')
                start = time.time()
                fut = client.map(lambda v: save_chunk(arr_src, dest_arr, v, invert), part)
                print(f'Submitted {len(part)} tasks to the scheduler in {time.time()- start}s')
                # wait for all the futures to complete
                result = wait(fut)
                print(f'Completed {len(part)} tasks in {time.time() - start}s')
                



