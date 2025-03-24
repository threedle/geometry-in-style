from typing import Sequence
import numpy as np
import sys
import os
import igl
import json

_PSREC_ARRAY_TAG = "PSREC_ARRAY"
psrec_fnames = sys.argv[1:]
for psrec_fname in psrec_fnames:
    dirname = os.path.dirname(psrec_fname)
    basename = os.path.basename(psrec_fname)
    basename_noext = os.path.splitext(basename)[0]
    savename = os.path.join(
        dirname, "reslt" + basename_noext.removeprefix("psrec") + ".obj"
    )
    print(savename)
    __is_psrec_tagged_array = (
        lambda a: isinstance(a, Sequence)
        and not isinstance(a, str)
        and len(a) == 2
        and a[0] == _PSREC_ARRAY_TAG
        and isinstance(a[1], int)
    )
    with np.load(psrec_fname) as npz:
        # look for the last register_surface_mesh call
        calls_concatenated = str(npz["calls"])
        args_with_arrays_as_PSREC_ARRAY_npzkey = None
        for call_serialized in calls_concatenated.split("\r"):
            lines = call_serialized.split("\n")
            if not ((n_lines := len(lines)) <= 3 and n_lines > 0):
                raise ValueError(
                    f"(this is a psrec parse error) error parsing serialized function call with lines {lines}"
                )
            # first "line" contains the function name. it is either just a string,
            # or it can be of the form f"{_PSREC_PROXY_TAG}!{registered_struct_proxy_id}.{function_name}"
            # or f"{_PSREC_PSPLT_TAG}v{pspltversion}!{function_name}"
            function_name = lines[0]
            if function_name != "register_surface_mesh":
                continue

            args = json.loads(lines[1])
            assert isinstance(args, Sequence) and not isinstance(args, str)
            # kwargs = json.loads(lines[2])
            # assert isinstance(kwargs, Sequence) and not isinstance(kwargs, str)
            args_are_psrec_tagged_array = tuple(map(__is_psrec_tagged_array, args))
            # kwargs_are_psrec_tagged_array = tuple(
            #     __is_psrec_tagged_array(kwarg) for _, kwarg in kwargs
            # )

            ## we've guaranteed the function is register_surface_mesh here...
            # so args[1] and args[2] will be the npz keys for the verts and faces arrays
            args_with_arrays_as_PSREC_ARRAY_npzkey = tuple(
                (f"{_PSREC_ARRAY_TAG}_{arg[1]}" if is_psrec_tagged_array else arg)
                for arg, is_psrec_tagged_array in zip(args, args_are_psrec_tagged_array)
            )

        # this will be the args for the LAST register_surface_mesh call when the loop ended
        # because we want to extract the final mesh verts and faces
        if args_with_arrays_as_PSREC_ARRAY_npzkey is not None:
            verts_npzkey = args_with_arrays_as_PSREC_ARRAY_npzkey[1]
            faces_npzkey = args_with_arrays_as_PSREC_ARRAY_npzkey[2]
            print(f"verts array key: {verts_npzkey}, faces array key:  {faces_npzkey}")
        else:
            raise ValueError("couldn't get args_with_arrays_as_PSREC_ARRAY_npzkey")

        verts = npz[verts_npzkey]
        faces = npz[faces_npzkey]
        igl.write_obj(savename, verts, faces)  # type: ignore
