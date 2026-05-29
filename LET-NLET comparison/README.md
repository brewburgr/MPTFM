# LET-NLET comparison environment

The analysis in the notebook `SurfaceNLET.ipynb` was performed inside a Docker container. We used the provided displacement data with the pre-defined parameters in the notebook for the Hertzian contact and Gaussian indenter profiles (varying the `Profile` parameter from `Hertzian` to `Gaussian`), and provide exemplary results in the according directory. The results contain integrated tractions, traction-indentation curves generated from them (using known strain/maximum indentation values), and ParaView files that can be used for visualization. 

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
