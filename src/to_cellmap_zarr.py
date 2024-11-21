import zarr
import os
import click
from numcodecs import Zstd
from pathlib import Path
import numpy as np
from dask.distributed import Client
from dask_jobqueue import LSFCluster
from dask.distributed import LocalCluster
import zarr_attrs_multiscales as to_ngff
#import copy_data_chunk_by_chunk as cd
#from copy_data import cluster_compute
from typing import Literal, Tuple, Union
import cellmap_layout as cml
from dask.array.core import slices_from_chunks, normalize_chunks
from toolz import partition_all
from dask.distributed import wait

import os
import zarr
import dask.array as da
import time



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
    client.cluster.scale(num_workers)

    for src_obj, dest_group in src_dest_info:

        # if isinstance(src_obj, zarr.core.Array):
        #     zarrays = [(src_obj.basename, src_obj)]
        # else:
        #     zarrays = src_obj.arrays(recurse = True)
        client.cluster.scale(num_workers)

        if isinstance(src_obj, zarr.core.Array):
            zarrays = [src_obj]
        else:
            zarrays = [key_val_arr[1] for key_val_arr in (src_obj.arrays())]
        
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



@click.command()
@click.option('--src', '-s', type=click.Path(exists = True))
@click.option("--dest", '-d', type=click.Path())
@click.option('--mtype', '-mt', default = "em", type=click.STRING)
@click.option('--gtruth', '-gt', default = "", type=click.STRING)
@click.option('--inf', '-i' , default = "", type=click.STRING)
@click.option('--masks', '-m', default = "", type=click.STRING)
@click.option('--lm', '-lm', default = "", type=click.STRING)
@click.option('--num_workers', '-c', default = 200, type=click.INT)
@click.option('--cluster', '-s', default = "lsf", type=click.STRING)
@click.option('--clevel', '-cl', default = 6, type=click.INT)
@click.option('--max_dask_chunk_num', '-maxchnum' , default = 50000, type=click.INT)
@click.option('--dry', default = False, type=click.BOOL)

def cli(src, dest, mtype, gtruth, inf, masks, lm, num_workers, cluster, clevel, max_dask_chunk_num, dry):
    compressor = Zstd(level=clevel)

    num_cores = 1
    cluster = LSFCluster(
        cores=num_cores,
        processes=num_cores,
        memory=f"{15 * num_cores}GB",
        ncpus=num_cores,
        mem=15 * num_cores,
        walltime="48:00",
        local_directory = "/scratch/$USER/"
        )
    
    #cluster = LocalCluster()
    client = Client(cluster)
    with open(os.path.join(os.getcwd(), "dask_dashboard_link" + ".txt"), "w") as text_file:
        text_file.write(str(client.dashboard_link))
    print(client.dashboard_link)


    #figure out the layout of an output .zarr file. 
    recon_groups = cml.get_store_info(src, mtype, inference = inf, groundtruth = gtruth, masks = masks, lm = lm)

    #copy groups and arrays info to an output zarr file.  
    root_dest, src_dest_info, zs = cml.create_cellmap_tree(recon_groups, dest, compressor)
    
    #add ome ngff multiscale metadata, if applicable
    for item in src_dest_info:
        if isinstance(item[0], zarr.hierarchy.Group): 
            dest_group = root_dest[item[1]]    
            to_ngff.normalize_to_ngff(dest_group)
    
    #copy input .n5 arrays data to corresponding arrays in the output .zarr file.
    
    
    if dry == False:
        copy_arrays_data(src_dest_info,
                            zs,
                            client,
                            num_workers,
                            invert=False,
                            comp=compressor)
        
        #copy_arrays_data(src_dest_info, zs, max_dask_chunk_num, compressor)
   

if __name__ == '__main__':
    cli()
