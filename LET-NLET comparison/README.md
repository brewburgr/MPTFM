# LET-NLET comparison environment

The analysis in the notebook `SurfaceNLET.ipynb` was performed inside a Docker container for reproducibility. We used the displacement data with the pre-defined parameters in the notebook for the Hertzian contact and Gaussian indenter profiles (varying the `Profile` parameter).

- **Docker image:** `dolfinx/dolfinx:stable`
- **Image date:** May 28, 2026
- **Exact image digest:** `sha256:25a4952542107766402a1de448ab1b387df70243e10f9d3f717d8c353f2acae4`

## Pull and run

```bash
# Reproducible: exact image
docker pull dolfinx/dolfinx@sha256:25a4952542107766402a1de448ab1b387df70243e10f9d3f717d8c353f2acae4
docker run -it --rm -p 8888:8888 dolfinx/dolfinx@sha256:25a4952542107766402a1de448ab1b387df70243e10f9d3f717d8c353f2acae4

# Or (may give a different version in the future)
docker pull dolfinx/dolfinx:stable
docker run -it --rm -p 8888:8888 dolfinx/dolfinx:stable
