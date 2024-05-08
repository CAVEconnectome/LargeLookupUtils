import cloudvolume
from multiwrapper import multiprocessing_utils as mu
import time
import numpy as np
from retrying import retry
import argparse
import os
import fastremap


@retry(
    wait_exponential_multiplier=1000, wait_exponential_max=10000, stop_max_delay=30000
)
def load_cv(cv_path, progress=False):
    return cloudvolume.CloudVolume(
        cv_path,
        progress=progress,
        bounded=False,
        fill_missing=True,
        use_https=True,
    )


def get_chunk_ids(cv, coords):
    """Extracts chunk ids from coordinates."""
    bbox = np.array(cv.bounds.to_list()).reshape(2, 3)
    return ((coords - bbox[0]) / cv.chunk_size).astype(int)


def sort_by_chunk(cv, coords):
    """Sorts coordinates by chunk ids."""
    chunk_ids = get_chunk_ids(cv, coords)
    sorting = np.argsort(fastremap.unique(chunk_ids, return_inverse=True, axis=0)[1])

    return sorting


def _lookup_sv_ids_chunk(args):
    """Lookup supervoxel IDs chunk by chunk"""
    coord_ids, coords, cv_path = args

    cv = load_cv(cv_path)

    chunk_ids = get_chunk_ids(cv, coords)

    sv_ids_block = []
    coord_ids_block = []

    for chunk_id in np.unique(chunk_ids, axis=0):
        chunk_m = np.all((chunk_ids - chunk_id) == 0, axis=1)

        chunk_coords = coords[chunk_m]

        min_coord = np.min(chunk_coords, axis=0)
        max_coord = np.max(chunk_coords, axis=0)

        ws_seg_chunk = cv[
            min_coord[0] : max_coord[0] + 1,
            min_coord[1] : max_coord[1] + 1,
            min_coord[2] : max_coord[2] + 1,
        ]

        chunk_coords = coords[chunk_m] - min_coord

        sv_ids = ws_seg_chunk[
            chunk_coords[:, 0], chunk_coords[:, 1], chunk_coords[:, 2]
        ].flatten()

        coord_ids_block.extend(coord_ids[chunk_m])
        sv_ids_block.extend(sv_ids)

    return coord_ids_block, sv_ids_block


def lookup_sv_ids(
    cv_path,
    synapse_coord_path,
    out_path,
    coord_sorting=None,
    scaling=None,
    n_processes=64,
):
    """Lookups of supervoxel IDs"""
    cv = load_cv(cv_path)

    if scaling is None:
        scaling = np.array(cv.resolution)
    else:
        assert len(scaling) == 3

    all_coords = np.load(synapse_coord_path)
    all_coords = (all_coords / scaling).astype(int)

    if coord_sorting is None:
        coord_sorting = sort_by_chunk(cv, all_coords)

    n_jobs = min(len(coord_sorting), n_processes * 30)

    multi_args = []
    coord_sorting_blocks = np.array_split(coord_sorting, n_jobs)
    for coord_sorting_block in coord_sorting_blocks:
        multi_args.append(
            [coord_sorting_block, all_coords[coord_sorting_block], cv_path]
        )

    time_start = time.time()
    if n_processes == 1:
        rs = mu.multithread_func(
            _lookup_sv_ids_chunk, multi_args, n_threads=1, debug=True
        )
    else:
        rs = mu.multisubprocess_func(
            _lookup_sv_ids_chunk,
            multi_args,
            n_threads=n_processes,
            package_name="largelookuputils",
        )

    print(f"TIME {time.time() - time_start}")

    coord_ids = []
    sv_ids_unordered = []
    for r in rs:
        coord_ids.extend(r[0])
        sv_ids_unordered.extend(r[1])

    coord_ids = np.array(coord_ids)
    sv_ids_unordered = np.array(sv_ids_unordered)

    np.save(f"{out_path}/sv_coord_ids.npy", coord_ids)
    np.save(f"{out_path}/sv_ids_unordered.npy", sv_ids_unordered)

    sv_ids_ordered = np.zeros(len(sv_ids_unordered), dtype=np.uint64)
    sv_ids_ordered[coord_ids] = sv_ids_unordered

    np.save(f"{out_path}/sv_ids_ordered.npy", sv_ids_ordered)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Lookup supervoxel IDs for a large number of coordinates"
    )

    parser.add_argument(
        "--synapse_coord_path",
        "-s",
        type=str,
        required=True,
        help="Synapse coordinates as Nx3 numpy file",
    )

    parser.add_argument(
        "--cloudvolume_path",
        "-c",
        type=str,
        required=True,
        help="Path for cloudvolume init",
    )

    parser.add_argument(
        "--out_path",
        "-o",
        type=str,
        required=True,
        help="Path for data dump",
    )

    parser.add_argument(
        "--processes",
        "-p",
        type=int,
        required=True,
        help="Number of processes",
    )

    args = parser.parse_args()
    print(args)

    assert os.path.exists(args.synapse_coord_path)

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    lookup_sv_ids(
        args.cloudvolume_path,
        args.synapse_coord_path,
        args.out_path,
        coord_sorting=None,
        n_processes=args.processes,
    )
