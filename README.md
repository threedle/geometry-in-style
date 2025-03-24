## Geometry in Style: 3D Stylization via Surface Normal Deformation

The official implementation for our CVPR 2025 paper.

### Environment setup
Run
```
conda env create -f environment-minimal-geomstyle.yml
```
> NOTE there is a small chance you may need **micromamba** instead of conda to solve this
environment in a reasonable amount of time. Your cluster may not have this installed; follow
[instructions here](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html) (I prefer the manual installation, putting the `micromamba` executable in your home directory somewhere if you don't have root permissions such as on a cluster.) Afterwards run conda commands using `micromamba` instead of `conda`.

### Instructions for generating a ton of configs and running `deform_with_csd_dARAP.py`

- Read and edit the variables at the top of `prep_sds_run_configs.sh` (up until the line `# END VARIABLES FOR CONFIG`) for this batch of runs
- Run `bash prep_sds_run_configs.sh`. This will open an interactive prompt where you will enter the prompt and shortname (a no-space nickname to describe the prompt-and-shape pair in the filename) for each mesh in the dataset.
- Once you've entered prompts and shortnames for every mesh in the set, you will have a bunch of config JSON files as reported by the printout.
- Use your preferred sbatch/job scheduling method to run each of these configs with `python deform_with_csd_dARAP.py -c CONFIGFILE`.

### Instructions to do one run
- Read `v3deform_with_csd_working_config-2024-11-04.json` the current latest config.
- Edit fields with your prompts, paths to your meshes, and run descriptions in filenames
- Polyscope recording and result save filenames must have a valid parent directory (the code will **not** automatically `mkdir` any parent directory structure to fulfill requested save paths!)
- To run, just do `python deform_with_csd_dARAP.py -c CONFIGFILE`
- On the cluster (where polyscope init is not available), export the environment variable `NO_POLYSCOPE=1` before running.

### Quick example run
(replace `/usr/local/cuda-12.1` with your system's installation of the cuda dev toolkit if not that)
```bash
NO_POLYSCOPE=1 CUDA_HOME=/usr/local/cuda-12.1/ python deform_with_csd_dARAP.py -c example-run/confg-example-cuteanimalthemedchair.json
```
- After this is done, you should get the recording file `example-run/psrec-deform-csdv3-example-cuteanimalthemedchair.npz` and a saved deformation quantity `example-run/nrmls-deform-csdv3-example-cuteanimalthemedchair.npz`.
- You can extract the final mesh out of the psrec recording using
```bash
python scratch_extract_final_mesh_from_psrec.py example-run/psrec-deform-csdv3-example-cuteanimalthemedchair.npz
```
  - (that should save `example-run/reslt-deform-csdv3-example-cuteanimalthemedchair.obj`. You can compare that result with `example-run/reslt-expected-cuteanimalthemedchair.obj`. We don't  fix the seed so there may be minor aesthetic differences but the result (namely the ears) should be mostly the same.)
- You can also play back the Polyscope recording (on a system with a display, not a headless server) with
```bash
python thlog.py replay psrec-deform-csdv3-example-cuteanimalthemedchair.npz
```
- (This will show the usual Polyscope window but with a "Playback controls" window. Step through the `show`-frames of the recording by clicking the button on the "Playback controls" window that shows up.)


### Misc repro notes
- We use pytorch3d's cotangent laplacian function which happens to use `cot a + cot b` weights rather than `0.5 * (cot a + cotb)` like libigl's `cotmatrix` function. This had no effect on our implementation of the ARAP global solve right-hand-side since the same weights are in the matrix and the right-hand-side construction, but when using the prefactored IGL `arap_rhs` right-hand-side constructor, a `2 *` correction is needed on the resulting `rhs` assuming no rescaling back to the source shape's bounding box diagonal extent.
  - In practice, since we rescale the solved deformed shape to keep the same bounding box diagonal length as the source shape's bounding box diagonal length, this doesn't matter.
  - This does, however, affect the scale of the `lambda` hyperparameter for the Procrustes local step. The `lambda` values we report and use are with respect to these `cot a + cot b` laplacian weights, not the `igl.cotmatrix` weights which would require a 0.5x adjustment to the lambda hyperparameter.
  - To keep parity with the pytorch3d cot laplacian that we've been using, the `deformations_MINIMAL.py` file (with no pytorch3d dependencies, for ease of use in other projects) will compute the laplacian with `-2 * igl.cotmatrix(v,f)`. (The `-` is also because igl's cotmatrix follows the negative-definite convention, but we need the positive definite matrix for the solve.)

- In the paper we use a fixed FOV of 60 degrees for all runs, but a FOV range of (55, 80) is also good and can lessen any "global shearing" effect of the deformation in some cases.

### Variants
- `deformations_MINIMAL.py` is a variant of `deformations_dARAP.py` with **minimal dependencies** for easy inclusion in your projects (specifically no dependency on `pytorch3d`; the only dependencies are `numpy`, `scipy`, `torch`, `cholespy`, and `libigl`). See the docstring at the top of the file for info and usage.

### Type stubs
- In the `cholespy`, `pytorch3d`, `igl` folders are type stubs (`.pyi` files) containing type signatures for the functions from those libraries we use (for a better experience with static type checkers). Feel free to use them in your own projects involving these libraries if you use a static type checker.