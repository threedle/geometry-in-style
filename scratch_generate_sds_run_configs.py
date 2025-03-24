import sys
import os
import json
import readline
from glob import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_shapes", type=str, nargs="+", help="source meshes")
parser.add_argument("-j", "--base_json", type=str, help="base json config")
parser.add_argument(
    "-k",
    "--count_i_from",
    type=int,
    help="start counting the {i} number in --run_name_template from this number",
)

parser.add_argument(
    "-n",
    "--run_name_template",
    type=str,
    help="a run name template (placeholders: {i} for 1-indexed number; {i0} for 0-indexed number; {s} for a shortname describing the shape+prompt combo)",
)
parser.add_argument(
    "-r", "--result_dir", type=str, help="directory where run results will be saved"
)
parser.add_argument(
    "-s",
    "--savetype_prefix",
    type=str,
    help="prefix used for saving the optim result of these runs (e.g. nrmls, jcbns, tmplt)",
)
parser.add_argument(
    "-o", "--json_output_dir", type=str, help="location to save these json files"
)
namespace = parser.parse_args()
run_name_template = namespace.run_name_template
result_dir = namespace.result_dir
savetype_prefix = namespace.savetype_prefix
json_output_dir = namespace.json_output_dir
count_i_from = namespace.count_i_from

for s in (run_name_template, result_dir, savetype_prefix, json_output_dir):
    if any(c.isspace() for c in s):
        raise ValueError("can't have space in these strings")


# folder = namespace.folder
# folder containing the source shapes
base_json = namespace.base_json
# copy everything else from base_json EXCEPT for the
# "ps_recording_save_fname", "dataset.lists", "deform_by_csd.optimized_quantity_save_fname" keys
# obj_fnames = sorted(glob(os.path.join(folder, "*.obj")))
obj_fnames = sorted(namespace.input_shapes)
with open(base_json) as base_json_f:
    thedict = json.load(base_json_f)
    assert isinstance(thedict, dict)

assert "dataset" in thedict
assert "deform_by_csd" in thedict

json_copies = []

result_fnames = []
# run_name_template = input(
#     "Input a run name template (placeholders: {i} for 1-indexed number; {i0} for 0-indexed number; {s} for a shortname describing the shape+prompt combo)\n> "
# )
# result_dir = input("Input directory where results will be saved:\n> ")
# savetype_prefix = input(
#     "Input the prefix used for saving the optim result of this run (e.g. nrmls, jcbns, tmplt):\n> "
# )
# json_dump_dir = input("Input the location to save these json files:\n> ")

last_nonempty_prompt = None
last_nonempty_shortname = None
prompts = []
shortnames = []
for i, obj_fname in enumerate(obj_fnames):
    dirname = os.path.dirname(obj_fname)
    basename_noext, ext = os.path.splitext(os.path.basename(obj_fname))

    # enter prompt
    if last_nonempty_prompt is not None:
        prompt = input(
            f"Enter prompt for shape {obj_fname}\n(leave empty to use last-input prompt: {last_nonempty_prompt}):\n> "
        )
        if prompt:
            last_nonempty_prompt = prompt
    else:
        prompt = input(f"Enter prompt for shape {obj_fname}:\n> ")
    if prompt:
        last_nonempty_prompt = prompt
    else:
        if last_nonempty_prompt is None:
            raise ValueError("first entered prompt cannot be empty")
        # use last-input prompt
        prompt = last_nonempty_prompt

    prompts.append(prompt)

    # enter shortname
    if last_nonempty_shortname is not None:
        shortname = input(
            f"Enter shortname for this prompt + shape combo (to put in the run name). e.g. robotcow\n(leave empty to use last-input shortname: {last_nonempty_shortname}):\n> "
        )
        if shortname:
            last_nonempty_shortname = shortname
    else:
        shortname = input(
            f"Enter shortname for this prompt + shape combo (to put in the run name). e.g. robotcow\n> "
        )
    if shortname:
        last_nonempty_shortname = shortname
    else:
        if last_nonempty_shortname is None:
            raise ValueError("first entered shortname cannot be empty")
        # use last-input shortname
        shortname = last_nonempty_shortname

    # remove any whitespace and replace with _
    shortname = "_".join(shortname.split())
    shortnames.append(shortname)

    thedict["dataset"]["lists"] = {
        "fnames": [obj_fname],
        # "cache_fnames": [os.path.join(dirname, basename_noext + "-preproc" + ext)],
        "prompts": [prompt],
        "prompts_negative": [None],
    }
    runname = run_name_template.format(
        i0=i + count_i_from, i=i + count_i_from + 1, s=shortname
    )
    thedict["ps_recording_save_fname"] = os.path.join(
        result_dir, "psrec-" + runname + ".npz"
    )
    result_fname = os.path.join(result_dir, savetype_prefix + "-" + runname + ".npz")
    result_fnames.append(result_fname)
    thedict["deform_by_csd"]["optimized_quantity_save_fname"] = result_fname
    json_copies.append((runname, json.dumps(thedict, indent=2)))
    # psrecs.append('ps_recording_save_fname: "out-deform-by-csd-v3/"')


for out_name, jsonstring in json_copies:
    out_full_fname = os.path.join(json_output_dir, out_name + ".json")
    print(f"======writing {out_full_fname}")
    with open(out_full_fname, "w") as f:
        f.write(jsonstring)
    # print(jsonstring)
# print(json.dumps(thedict, indent=2))

# for convenience, also dump & prettyprint two lists: a list of source filenames, and a list of the to-be-made result filenames
with open(os.path.join(json_output_dir, "lists_for_hardsup.txt"), "w") as f:
    json.dump(
        {
            "optimize_by_mesh_dataset_io": {
                "patient_fnames": obj_fnames,
                "target_fnames": result_fnames,
            }
        },
        f,
        indent=2,
    )

# write the prompts and shortnames so we can just pipe it in stdin for an identical batch
with open(os.path.join(json_output_dir, "prompts_and_shortnames.txt"), "w") as f:
    for _prompt, _shortname in zip(prompts, shortnames):
        f.write(f"{_prompt}\n{_shortname}\n")
