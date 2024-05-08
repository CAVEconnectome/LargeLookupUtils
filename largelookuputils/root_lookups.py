import caveclient
import numpy as np
import os
import time
from multiwrapper import multiprocessing_utils as mu
import argparse


def _lookup_root_ids_thread(args):
    """Lookups of root IDs - thread"""
    datastack, sv_ids, coord_ids, i_thread, materialization_version = args

    client = caveclient.CAVEclient(datastack)
    timestamp = client.materialize.get_timestamp(materialization_version)

    root_ids = []
    lookedup_coord_ids = []
    id_blocks = np.array_split(np.arange(len(sv_ids)), int(len(sv_ids) / 10000) + 1)
    for i_num, id_block in enumerate(id_blocks):
        print(f"{i_thread}-{i_num + 1}/{len(id_blocks)}")

        sv_id_block = sv_ids[id_block]
        coord_id_block = coord_ids[id_block]

        sv_id_m = sv_id_block != 0

        sv_id_block = sv_id_block[sv_id_m]
        coord_id_block = coord_id_block[sv_id_m]

        root_ids.extend(client.chunkedgraph.get_roots(sv_id_block, timestamp=timestamp))
        lookedup_coord_ids.extend(coord_id_block)

    return root_ids, lookup_root_ids


def lookup_root_ids(
    datastack, in_path, out_path, materialization_version, n_processes=10
):
    """Lookups of root IDs"""
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    with open(f"{out_path}/materialization_version.txt", "w") as f:
        f.write(f"{materialization_version}")

    sv_coord_ids = np.load(f"{in_path}/sv_coord_ids.npy").flatten()
    sv_ids_unordered = np.load(f"{in_path}/sv_ids_unordered.npy").flatten()

    id_blocks = np.array_split(np.arange(len(sv_ids_unordered)), n_processes * 30)

    multi_args = []
    for i_num, id_block in enumerate(id_blocks):
        multi_args.append(
            [
                datastack,
                sv_ids_unordered[id_block],
                sv_coord_ids[id_block],
                i_num,
                materialization_version,
            ]
        )

    time_start = time.time()
    if n_processes == 1:
        rs = mu.multithread_func(
            _lookup_root_ids_thread, multi_args, n_threads=1, debug=True
        )
    else:
        rs = mu.multisubprocess_func(
            _lookup_root_ids_thread,
            multi_args,
            n_threads=n_processes,
            package_name="largelookuputils",
        )
    print(f"TIME {time.time() - time_start}")

    root_ids_unordered = []
    for r in rs:
        root_ids_unordered.extend(r[0])
        coord_ids.extend(r[1])

    coord_ids = np.array(coord_ids)
    root_ids_unordered = np.array(root_ids_unordered)

    np.save(f"{out_path}/root_ids_unordered.npy", root_ids_unordered)
    np.save(f"{out_path}/root_coord_ids.npy", coord_ids)

    root_ids_ordered = np.zeros(len(sv_ids_unordered), dtype=np.uint64)
    root_ids_ordered[coord_ids] = root_ids_unordered

    np.save(f"{out_path}/root_ids_ordered.npy", root_ids_ordered)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Lookup supervoxel IDs for a large number of coordinates"
    )

    parser.add_argument(
        "--sv_path",
        "-s",
        type=str,
        required=True,
        help="Path to numpy file with sv ids",
    )

    parser.add_argument(
        "--datastack",
        "-d",
        type=str,
        required=True,
        help="Datastack for lookups",
    )

    parser.add_argument(
        "--materialization_version",
        "-m",
        type=int,
        required=True,
        help="Materialization version for root lookup",
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

    assert os.path.exists(args.sv_path)

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    lookup_root_ids(
        args.datastack,
        args.sv_path,
        args.out_path,
        args.materialization_version,
        n_processes=args.processes,
    )
