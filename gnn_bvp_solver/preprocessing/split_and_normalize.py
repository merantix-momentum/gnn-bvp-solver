from functools import partial
from typing import Dict, Callable
from squirrel.driver.msgpack import MessagepackDriver
from squirrel.serialization import MessagepackSerializer
from squirrel.store import SquirrelStore
from squirrel.iterstream import IterableSource, Composable

import numpy as np


N_SAMPLES = 2500
MAX_VALUE = 10.0

SPLIT_25 = int(N_SAMPLES * 0.25)
SPLIT_50 = int(N_SAMPLES * 0.5)
SPLIT_80 = int(N_SAMPLES * 0.8)
SPLIT_90 = int(N_SAMPLES * 0.9)

N_SHARD = 100


def update_range_dict(range_dict: Dict, name: str, value: np.array, op: Callable = np.maximum) -> None:
    """Track maximum and minimum values for normalization"""
    if name in range_dict:
        range_dict[name] = op(value, range_dict[name])
    else:
        range_dict[name] = value


def unify_range_dicts(range_dict1: Dict, range_dict2: Dict, op: Callable = np.maximum) -> Dict:
    """Unify maximum and minimum values"""
    result = {}

    for name in range_dict1:
        result[name] = op(range_dict1[name], range_dict2[name])

    return result


def map_update_ranges(sample: Dict, range_dict: Dict) -> Dict:
    """Iterate samples and update minimums and maximums"""
    max_x = np.amax(np.abs(sample["data_x"]), axis=0)
    max_y = np.amax(np.abs(sample["data_y"]), axis=0)

    update_range_dict(range_dict, "x_range", max_x)
    update_range_dict(range_dict, "y_range", max_y)

    return sample


def get_range_dict(base_url: str, split: str) -> Dict:
    """Get maximums and minimums for normalization"""
    range_dict = {}

    it = MessagepackDriver(f"{base_url}/{split}").get_iter()
    it.map(partial(map_update_ranges, range_dict=range_dict)).tqdm().join()

    return range_dict


def save_shard(it: Composable, store: SquirrelStore) -> None:
    """Save set of shards"""
    store.set(value=list(it))


def scale(sample: Dict, range_dict: Dict) -> Dict:
    """Normalize example using the extreme values"""
    range_x = np.clip(range_dict["x_range"], a_min=0.000001, a_max=None)
    range_y = np.clip(range_dict["y_range"], a_min=0.000001, a_max=None)

    return {
        "data_x": sample["data_x"] / range_x.reshape(1, -1),
        "data_y": sample["data_y"] / range_y.reshape(1, -1),
        "edge_index": sample["edge_index"],
    }


def filter_max(sample: Dict) -> bool:
    """Filter outliers"""
    if sample["data_x"].max() > MAX_VALUE:
        return False
    if sample["data_y"].max() > MAX_VALUE:
        return False
    return True


def save_stream(
    it: Composable, output_url: str, split: str, range_dict: Dict = None, filter_outliers: bool = True
) -> None:
    """Scale, filter outliers and save composable as shards"""
    if it is None:
        return

    store = SquirrelStore(f"{output_url}/{split}", serializer=MessagepackSerializer())

    if range_dict is not None:
        it = it.map(partial(scale, range_dict=range_dict))

    if filter_outliers:
        it = it.filter(filter_max)

    it.batched(N_SHARD, drop_last_if_not_full=False).map(partial(save_shard, store=store)).tqdm().join()


def iterate_source_data(fem_generator: str) -> None:
    """Filter data for a single generator and iterate if necessary to create splits"""

    mesh_generators = [
        "square",
        "disk",
        "cylinder",
        "l_mesh",
        "u_mesh",
        "square_extra",
        "disk_extra",
        "cylinder_extra",
        "l_mesh_extra",
        "u_mesh_extra",
        "square_rand",
        "disk_rand",
        "cylinder_rand",
        "l_mesh_rand",
        "u_mesh_rand",
    ]

    for mesh_g in mesh_generators:
        key = f"{fem_generator}_{mesh_g}"
        path = f"gs://squirrel-core-public-data/gnn_bvp_solver/{key}"
        iter = MessagepackDriver(path).get_iter()

        print("GENERATING:", fem_generator, mesh_g)

        if mesh_g.startswith("u_mesh"):
            if mesh_g == "u_mesh":
                # test set 2
                # TRAIN1, VAL1, TRAIN2, VAL2, TEST1, TEST2
                yield None, None, None, None, None, iter
        else:
            # all but U-mesh
            if mesh_g.endswith("extra"):
                all_data = iter.tqdm().collect()

                # test set 1
                # TRAIN1, VAL1, TRAIN2, VAL2, TEST1, TEST2
                yield None, None, None, None, IterableSource(all_data[:SPLIT_25]), None
            elif mesh_g.endswith("rand"):
                all_data = iter.tqdm().collect()

                # train/val set 2
                # TRAIN1, VAL1, TRAIN2, VAL2, TEST1, TEST2
                yield None, None, IterableSource(all_data[:SPLIT_80]), IterableSource(all_data[SPLIT_80:]), None, None
            else:
                all_data = iter.tqdm().collect()

                # train/val set 1
                # TRAIN1, VAL1, TRAIN2, VAL2, TEST1, TEST2
                yield IterableSource(all_data[:SPLIT_80]), IterableSource(all_data[SPLIT_80:]), None, None, None, None


def scale_and_store(in_split: str, out_split: str, range_dict: Dict, base_url_in: str, base_url_out: str) -> None:
    """Normalize one stream and save it"""
    it = MessagepackDriver(f"{base_url_in}/{in_split}").get_iter()
    save_stream(it, base_url_out, out_split, range_dict)


def main(fem_generator: str, out_url: str) -> None:
    """Generate split for a single generator"""
    for append_train1, append_val1, append_train2, append_val2, append_test1, append_test2 in iterate_source_data(
        fem_generator
    ):
        print("saving splits")

        print("train1")
        save_stream(append_train1, out_url, "raw_train1")

        print("val1")
        save_stream(append_val1, out_url, "raw_val1")

        print("train2")
        save_stream(append_train2, out_url, "raw_train2")

        print("val2")
        save_stream(append_val2, out_url, "raw_val2")

        print("test1")
        save_stream(append_test1, out_url, "raw_test1")

        print("test2")
        save_stream(append_test2, out_url, "raw_test2")

        print("moving on")


def main_scale(in_url: str, out_url: str) -> None:
    """Apply normalization to generated data"""
    range_dict1 = get_range_dict(in_url, "raw_train1")
    range_dict2 = get_range_dict(in_url, "raw_train2")
    range_dict = unify_range_dicts(range_dict1, range_dict2)

    print("unnormalized ranges: ", range_dict)
    print("scale and store")

    print("train")
    scale_and_store("raw_train1", "norm_train_no_ma", range_dict, in_url, out_url)
    scale_and_store("raw_train2", "norm_train_ma", range_dict, in_url, out_url)

    print("val")
    scale_and_store("raw_val1", "norm_val_no_ma", range_dict, in_url, out_url)
    scale_and_store("raw_val2", "norm_val_ma", range_dict, in_url, out_url)

    print("test1")
    scale_and_store("raw_test1", "norm_test_sup", range_dict, in_url, out_url)

    print("test2")
    scale_and_store("raw_test2", "norm_test_shape", range_dict, in_url, out_url)


def process(generator_key: str) -> None:
    """Process data from a single fem generator"""
    base_url_gs = f"gs://squirrel-core-public-data/gnn_bvp_solver/{generator_key}"
    base_url = f"data/{generator_key}"  # store intermediate results locally

    main(generator_key, base_url)
    main_scale(base_url, base_url_gs)


if __name__ == "__main__":
    for label_g in ["ElectricsRandomChargeGenerator", "MagneticsRandomCurrentGenerator", "ElasticityFixedLineGenerator"]:
        process(label_g)
