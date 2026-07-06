from __future__ import annotations

import os

from autosat_core.common import (
    clean_files,
    collect_results,
    collect_results_eval,
    copy_folder,
    delete_InfiniteLoopInst,
    find_key_for_value,
    get_batch_id,
    get_code,
    get_results_root,
    get_temp_root,
    revise_file,
)


def fill_core_codes(origin_file, target_file, answer_code, **kwargs):
    timeout_value = kwargs.pop("timeout", "{{ timeout }}")
    data_dir_value = kwargs.pop("data_dir", "{{ data_dir }}")
    revise_file(
        file_name=origin_file,
        save_dir=target_file,
        timeout=timeout_value,
        data_dir=data_dir_value,
        replace_code=answer_code,
        **kwargs,
    )


def train_init(args):
    temp_root = get_temp_root()
    results_folder = "./temp/results/"
    prompts_folder = os.path.join(temp_root, "prompts")
    easy_sat_folder = os.path.join(temp_root, "EasySAT")

    os.makedirs(results_folder, exist_ok=True)
    os.makedirs(prompts_folder, exist_ok=True)

    if os.path.exists(results_folder):
        clean_files(folder_path=results_folder, mode="all")
    else:
        os.makedirs(results_folder)
    copy_folder("solver/baseline", args.batch_size, mode="eval", target_folder=easy_sat_folder)
    clean_files(folder_path=easy_sat_folder, mode="exe")
    copy_folder(easy_sat_folder, args.batch_size)
    return


def check_reIteration(round, best_result_dict, baseline):
    if round != 1:
        return False
    best_results = next(iter(best_result_dict.values()))
    if best_results[0] < baseline["time"] or best_results[2] < baseline["PAR-2"]:
        return False
    return True


__all__ = [
    "clean_files",
    "collect_results",
    "collect_results_eval",
    "copy_folder",
    "delete_InfiniteLoopInst",
    "find_key_for_value",
    "fill_core_codes",
    "get_batch_id",
    "get_code",
    "get_results_root",
    "get_temp_root",
    "revise_file",
    "train_init",
    "check_reIteration",
]
