# LET-NLET comparison environment

The analysis in the notebook `SurfaceNLET.ipynb` was performed inside a Docker container. We used the provided displacement data with the pre-defined parameters in the notebook for the Hertzian contact, Gaussian indenter, and ring indenter profiles, and provide example results in the according directory. For each scenario, results include integrated tractions and curves comparing results by linear elasticity theory (LET) and non-linear elasticity theory (NLET, neo-Hookean model) generated from them (using known strain/maximum indentation values), which are also shown in the notebook. ParaView files for visualization are created within the workflow as well.

- **Docker image:** `dolfinx/dolfinx:stable`
- **Image date:** May 28, 2026
- **Exact image digest:** `sha256:25a4952542107766402a1de448ab1b387df70243e10f9d3f717d8c353f2acae4`

## Pull and run

```bash
# Reproduce exact image:
docker pull dolfinx/dolfinx@sha256:25a4952542107766402a1de448ab1b387df70243e10f9d3f717d8c353f2acae4
docker run -it --rm -p 8888:8888 dolfinx/dolfinx@sha256:25a4952542107766402a1de448ab1b387df70243e10f9d3f717d8c353f2acae4

# Or (may give a different version in the future):
docker pull dolfinx/dolfinx:stable
docker run -it --rm -p 8888:8888 dolfinx/dolfinx:stable
```

## Workflow overview

1. Build a spherical mesh with Gmsh.
2. Map the reconstructed surface displacement field onto the mesh as a Dirichlet boundary condition.
3. Solve the interior displacement field with FEniCSx for both LET and NLET (neo-Hookean) constitutive laws.
4. Integrate the resulting traction on the indenter surface and compare the integrated-traction-indentation curves.
