Windows program from https://robust-low-poly-meshing.github.io/.

Can be run on Ubuntu with wine:
```sh
wine SurfaceRemeshingCli_bin.exe -o out_folder -n 100 -i input_mesh.obj
```
Arguments:
```
Positionals:
  input TEXT:FILE REQUIRED    input mesh.
  input INT REQUIRED          screen size.

Options:
  -h,--help                   Print this help message and exit
  -i,--inputMesh TEXT:FILE REQUIRED
                              input mesh.
  -n,--screenSize INT REQUIRED
                              screen size.
  -f,--finalFaceNum INT       final face number: If given, the final #face <= g.
  -o,--savingFolder TEXT      saving folder, optional, if not given, set to wor/
  -s,--shell                  Make the shell of the input mesh, optional, defaue
  -t,--triFanCheckMethod INT  the method used to check self-intersection between
  -p,--timingProfile          whether profile the flip and collapse time in detl
```

## License Notice

The RoLoPoly binary (`SurfaceRemeshingCli_bin.exe`) is redistributed from the
[Robust Low-Poly Meshing](https://robust-low-poly-meshing.github.io/) project by
Zhen Chen et al. (SIGGRAPH 2023). See that website for the licensing terms of this software.
