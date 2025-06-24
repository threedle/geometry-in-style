## Geometry in Style: 3D Stylization via Surface Normal Deformation

The official implementation for our CVPR 2025 paper.

[paper](https://people.cs.uchicago.edu/~namanh/papers/geometry-in-style.pdf)
| [project page](https://threedle.github.io/geometry-in-style/)
| [arXiv](http://arxiv.org/abs/2503.23241)
| [use in your projects!](#minimal-dependency-deformation-code-for-your-projects)
| [bibtex](#bibtex)

![figure with 4 frames. top left is gray text 'a 3d render of'. underneath each frame, blue text is the rest of a prompt. The blue text lines are "a pineapple-themed vase", "an A-pose knight in armor", "a cute animal-themed chair", "a lego goat". above each prompt, a gray unmodified shape shows the source mesh (a vase, an A-pose human, a rocking chair, a goat), and the large blue mesh is the source shape deformed towards the style prompt in a vivid, detailed, but identity-preserving way.](assets/teaser.png)

### Environment setup
Run this on a system with a GPU (i.e. `$ nvidia-smi` works)
```
conda env create -f environment-minimal-geomstyle.yml
```
> NOTE there is a small chance you may need **micromamba** instead of conda to solve this
environment in a reasonable amount of time. Your cluster may not have this installed; follow
[instructions here](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html) (I prefer the manual installation, putting the `micromamba` executable in your home directory somewhere if you don't have root permissions such as on a cluster.) Afterwards run conda commands using `micromamba` instead of `conda`.

Optimizations were tested to work on an NVIDIA A40 (48GB) GPU. Your GPU should
ideally have capacity to hold both `DeepFloyd/IF-I-XL-v1.0` and
`DeepFloyd/IF-II-L-v1.0` models, though configuring `--deform_by_csd.guidance.cpu_offload true`
may help with lower memory capacities.

Either your CPU or a much less powerful GPU can be used to run `apply_saved_deform_qty_npz.py`
which applies a saved deformation (a `nrmls-*.npz` file) to its associated mesh, with a tunable λ
hyperparameter.

No discrete GPU is required nor used for playing back `psrec-*.npz`
Polyscope recording files of results, as long as the playback machine has a
screen (so you can run it on your local machine, but not on a headless system.)


**To set up HuggingFace Hub and your account for downloading the DeepFloyd stages (instructions from [DeepFloyd IF](https://github.com/deep-floyd/IF)):**

1) If you do not already have one, create a [Hugging Face account](https://huggingface.co/join)
2) Accept the license on the model card of [DeepFloyd/IF-I-XL-v1.0](https://huggingface.co/DeepFloyd/IF-I-XL-v1.0)
3) Log in to Hugging face locally. In the conda environment you just created, install `huggingface_hub`
```
pip install huggingface_hub --upgrade
```
run the login function in a python shell
```
from huggingface_hub import login

login()
```
and enter your [Hugging Face Hub access token](https://huggingface.co/docs/hub/security-tokens#what-are-user-access-tokens).

### Minimal-dependency deformation code (for your projects)
`variants/deformations_MINIMAL.py` is a self-contained variant of `deformations_dARAP.py` with **minimal dependencies** for easy inclusion in your projects (specifically no dependency on `pytorch3d`; the only dependencies are `numpy`, `scipy`, `torch`, `cholespy`, and `libigl`). See the docstring at the top of the file for info and usage.

Usage in a nutshell:
(check `deformations_MINIMAL.py` for more information.)

```python
# supports batching. for example, operating on a batch of two meshes (mesh1, mesh2) not necessarily of the same vertex and face counts
(verts1, faces1) = mesh1
(verts2, faces2) = mesh2
verts_list = [verts1, verts2]
faces_list = [faces1, faces2]

# prepare the solver
solvers = SparseLaplaciansSolvers.from_meshes(
  verts_list, faces_list,
  pin_first_vertex=True,
  compute_poisson_rhs_lefts=False,
  compute_igl_arap_rhs_lefts=None,
)

# find per-vertex matrices with local step
procrustes_precompute = ProcrustesPrecompute.from_meshes(
    local_step_procrustes_lambda=lamb,
    arap_energy_type="spokes_and_rims_mine", 
    laplacians_solvers=solvers, 
    verts_list=verts_list, 
    faces_list=faces_list
)
per_vertex_3x3matrices_packed = calc_rot_matrices_with_procrustes(
    procrustes_precompute, 
    torch.cat(verts_list), 
    torch.cat(target_normals_list)
)
# (assuming target_normals_list is a list of tensors each (n_verts, 3), each tensor is the target per-vertex normals of a single mesh in the batch)

per_vertex_3x3matrices_list = torch.split(per_vertex_3x3matrices_packed, [len(v) for v in verts_list])

# solve for deformations given per-vertex 3x3 matrices
deformed_verts_list = calc_ARAP_global_solve(
    verts_list, faces_list, solvers, per_vertex_3x3matrices_list,
    arap_energy_type="spokes_and_rims_mine",
    postprocess="recenter_only"
)
```

### Instructions for generating a ton of configs and running `deform_with_csd_dARAP.py`

- Read and edit the variables at the top of `prep_sds_run_configs.sh` (up until the line `# END VARIABLES FOR CONFIG`) for this batch of runs
- Run `bash prep_sds_run_configs.sh`. This will open an interactive prompt where you will enter the prompt and shortname (a no-space nickname to describe the prompt-and-shape pair in the filename) for each mesh in the dataset.
- Once you've entered prompts and shortnames for every mesh in the set, you will have a bunch of config JSON files as reported by the printout; each JSON file is the configuration for one run.
- Use your preferred sbatch/job scheduling method to run each of these configs with `python deform_with_csd_dARAP.py -c CONFIGFILE` where `CONFIGFILE` is the JSON config filename for the run.

### Instructions to do one run
- Read `confg-base.json` the current latest base config.
- Edit fields with your prompts, paths to your meshes, and run descriptions in filenames
- Polyscope recording and result save filenames must have a valid parent directory (the code will **not** automatically `mkdir` any parent directory structure to fulfill requested save paths!)
- To run, just do `python deform_with_csd_dARAP.py -c confg-base.json` (or whatever your config filename is)
- On the cluster (where polyscope init is not available), export the environment variable `NO_POLYSCOPE=1` before running.
> Overriding fields on the command line is supported.
> For instance, you can do `python deform_with_csd_dARAP.py -c CONFIGFILE --deform_by_csd.n_iters 1000` to override the `deform_by_csd.n_iters` field to be 1000 instead of the value in `CONFIGFILE`.

### Quick example run: the *cute animal-themed chair*
(replace `/usr/local/cuda-12.1` with your system's installation of the cuda dev toolkit if not that)
```bash
NO_POLYSCOPE=1 CUDA_HOME=/usr/local/cuda-12.1/ python deform_with_csd_dARAP.py -c example-run/confg-example-cuteanimalthemedchair.json
```
- After this is done, you should get the recording file `example-run/psrec-deform-csdv3-example-cuteanimalthemedchair.npz` and a saved deformation quantity `example-run/nrmls-deform-csdv3-example-cuteanimalthemedchair.npz`.
- You can **extract the final mesh** out of the psrec recording using
```bash
python scratch_extract_final_mesh_from_psrec.py example-run/psrec-deform-csdv3-example-cuteanimalthemedchair.npz
```
  - (that should save `example-run/reslt-deform-csdv3-example-cuteanimalthemedchair.obj`. You can compare that result with `example-run/reslt-expected-cuteanimalthemedchair.obj`. We don't  fix the seed so there may be minor aesthetic differences but the result (namely the ears) should be mostly the same.)
- You can also **play back the Polyscope recording** (on a system with a display, not a headless server) with
```bash
python thlog.py replay psrec-deform-csdv3-example-cuteanimalthemedchair.npz
```
- (This will show the usual Polyscope window but with a "Playback controls" window. Step through the `show`-frames of the recording by clicking the button on the "Playback controls" window that shows up.)

- **To apply the saved deformation** (an optimized set of normals, saved alongside the original source mesh) such as the provided `nrmls-expected-cuteanimalthemedchair.npz` or the `nrmls-deform-csdv3-example-cuteanimalthemedchair.npz` file that is saved after the example optimization run, simply do
```bash
python apply_saved_deform_qty_npz.py example-run/nrmls-expected-cuteanimalthemedchair.npz
```
- You can also add `--cpu true` to run on a local, CPU-only machine; you can add `--lamb x` to set a different lambda hyperparameter (deformation strength) with which to apply this saved deformation (higher is stronger.) The saved lambda used during optimization is 8.0; try setting it to 5 or 10.

### Misc repro notes
- We use pytorch3d's cotangent laplacian function which happens to use `cot a + cot b` weights rather than `0.5 * (cot a + cotb)` like libigl's `cotmatrix` function. This had no effect on our implementation of the ARAP global solve right-hand-side since the same weights are in the matrix and the right-hand-side construction, but when using the prefactored IGL `arap_rhs` right-hand-side constructor, a `2 *` correction is needed on the resulting `rhs` assuming no rescaling back to the source shape's bounding box diagonal extent.
  - In practice, since we rescale the solved deformed shape to keep the same bounding box diagonal length as the source shape's bounding box diagonal length, this doesn't matter.
  - This does, however, affect the scale of the `lambda` hyperparameter for the Procrustes local step. The `lambda` values we report and use are with respect to these `cot a + cot b` laplacian weights, not the `igl.cotmatrix` weights which would require a 0.5x adjustment to the lambda hyperparameter.
  - To keep parity with the pytorch3d cot laplacian that we've been using, the `deformations_MINIMAL.py` file (with no pytorch3d dependencies, for ease of use in other projects) will compute the laplacian with `-2 * igl.cotmatrix(v,f)`. (The `-` is also because igl's cotmatrix follows the negative-definite convention, but we need the positive definite matrix for the solve.)

- In the paper we use a fixed FOV of 60 degrees for all runs, but a FOV range of (55, 80) is also good and can lessen any "global shearing" effect of the deformation in some cases.

### Type stubs
- In the `cholespy`, `pytorch3d`, `igl` folders are type stubs (`.pyi` files) containing type signatures for the functions from those libraries we use (for a better experience with static type checkers). Feel free to use them in your own projects involving these libraries if you use a static type checker.

### BibTeX
```bibtex
@InProceedings{Dinh_2025_CVPR,
    author    = {Dinh, Nam Anh and Lang, Itai and Kim, Hyunwoo and Stein, Oded and Hanocka, Rana},
    title     = {Geometry in Style: 3D Stylization via Surface Normal Deformation},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {28456-28467}
}
```