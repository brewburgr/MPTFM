import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pyshtools as _psh

import scipy.sparse as _spm
from scipy.io import loadmat, savemat

from shelastic.shelastic import calUmode, calSmode
from shelastic.shutil import SHCilmToVector, CartCoord_to_SphCoord, K2lmk


def rts(x, p=3):
    return float(np.format_float_scientific(x, precision=p, unique=False, trim="k"))


def nu(m, l):
    return l / 2 / (l + m)


def E(m, l):
    return m * (3 * l + 2 * m) / (l + m)


def distance(a, b):
    l = len(a)
    return np.sqrt(sum([(a[n] - b[n]) ** 2 for n in range(l)]))


def lame_mu(E, nu):
    return E / 2 / (1 + nu)


def lame_lambda(E, nu):
    return E * nu / (1 + nu) / (1 - 2 * nu)


def vector_sph_to_cart(vector, theta, phi):
    A = np.array(
        [
            [np.sin(theta) * np.cos(phi), np.cos(theta) * np.cos(phi), -np.sin(phi)],
            [np.sin(theta) * np.sin(phi), np.cos(theta) * np.sin(phi), np.cos(phi)],
            [np.cos(theta), -np.sin(theta), 0],
        ]
    )
    return A.dot(vector)


def vector_cart_to_sph(vector, theta, phi):
    A = np.array(
        [
            [np.sin(theta) * np.cos(phi), np.cos(theta) * np.cos(phi), -np.sin(phi)],
            [np.sin(theta) * np.sin(phi), np.cos(theta) * np.sin(phi), np.cos(phi)],
            [np.cos(theta), -np.sin(theta), 0],
        ]
    ).transpose()
    return A.dot(vector)


def cart_to_sph_coords(vector):
    # input [x,y,z]
    # output [r,theta,phi]
    x, y, z = vector
    r = np.linalg.norm(vector)
    phi = np.arctan2(y, x)
    if z == 0:
        theta = np.pi / 2
    else:
        theta = np.arccos(z / r)
    if phi < 0:
        phi += 2 * np.pi
    return np.array([r, theta, phi])


def all_errors_2D_old(
    x_true,
    x_est,
    type_=None,
    boxsize=1.2,
    radius=1,
    hist=True,
    treshold=1e-6,
    thetarange=[0, np.pi],
):
    """
    Parameters
    type: 'slice' or 'surface'
    """
    dimx, dimz = np.shape(x_true)
    # thetas=np.linspace(*thetarange,dimz)
    thetas0, _ = _psh.expand.GLQGridCoord(dimz - 1)
    thetas = np.deg2rad(90 - thetas0)
    theta_matrix = np.kron(np.ones(dimx), thetas).reshape((dimx, dimz))
    x_true_weighted = np.multiply(x_true, theta_matrix)
    x_est_weighted = np.multiply(x_est, theta_matrix)
    n_weighted = dimx * dimz / np.pi
    if type_ == "slice":
        x_true = x_true_weighted
        x_est = x_est_weighted
        npoints = n_weighted
    else:
        npoints = dimx * dimz
    if hist:
        histdata = np.zeros((dimx, dimz))

    for nx in range(dimx):
        for nz in range(dimz):
            if type_ == "slice":
                val = (
                    (2 * abs(nx - dimx / 2 + 0.5) / dimx) ** 2
                    + (2 * abs(nz - dimz / 2 + 0.5) / dimz) ** 2
                ) * (boxsize / radius)
                if val > 1:
                    x_true[nx][nz] = 0
                    x_est[nx][nz] = 0
            #            if type_=='surface':

            if hist:
                x_t = x_true[nx][nz]
                x_e = x_est[nx][nz]
                if not np.isclose(x_t, 0, atol=treshold) and not np.isclose(
                    x_e, 0, atol=treshold
                ):
                    ae = abs(x_t - x_e)
                    rae = ae / x_t
                    histdata[nx, nz] = rae

    # n_nonzero=dimx*dimz-np.count_nonzero(np.isclose(x_est,np.zeros_like(x_est),atol=treshold))

    AE = np.sum(np.abs(x_est - x_true))
    MAE = AE / npoints
    MAE_rel = AE / np.sum(np.abs(x_true))

    SD = np.sum((x_est - x_true) ** 2)
    mean_abs = np.mean(np.abs(x_est))
    RMSD = np.sqrt(SD / npoints)
    NRMSD = RMSD / mean_abs
    if hist:
        return MAE, MAE_rel, RMSD, NRMSD, mean_abs, histdata
    else:
        return MAE, MAE_rel, RMSD, NRMSD, mean_abs


def AAD_surface(x):
    dimx, dimz = np.shape(x)
    thetas0, _ = _psh.expand.GLQGridCoord(dimx - 1)
    thetas = np.deg2rad(90 - thetas0)
    weights = np.sin(thetas)
    weightmatrix = np.kron(np.ones(dimz), weights).reshape((dimx, dimz))
    npoints = dimz * sum(weights)
    AAD = np.sum(np.multiply(np.abs(x), weightmatrix)) / npoints
    return AAD


def all_errors_2D(
    x_true,
    x_est,
    type_=None,
    boxsize=1.2,
    radius=1,
    hist=True,
    treshold=1e-6,
    thetarange="GLQ",
    customthetas=None,
    AAD2=0,
):
    """
    Parameters
    type: 'slice' or 'surface'
    """
    dimx, dimz = np.shape(x_true)
    # assert dimx < dimz
    if thetarange == "GLQ":
        thetas0, _ = _psh.expand.GLQGridCoord(dimx - 1)
        thetas = np.deg2rad(90 - thetas0)
        weights = np.sin(thetas)
    if thetarange == "custom":
        thetas = customthetas
    theta_matrix = np.kron(np.ones(dimz), weights).reshape((dimx, dimz))
    # x_true_weighted=np.multiply(x_true, theta_matrix)
    # x_est_weighted=np.multiply(x_est, theta_matrix)
    n_weighted = dimz * sum(weights)

    if type_ == "surface":
        # x_true=x_true_weighted
        # x_est=x_est_weighted
        npoints = n_weighted
        weightmatrix = theta_matrix
    elif type_ == "slice":
        npoints = 0
        weightmatrix = np.ones_like(x_true)
    else:
        npoints = dimx * dimz
    if hist:
        histdata = np.zeros((dimx, dimz))

    for nx in range(dimx):
        for nz in range(dimz):
            if type_ == "slice":
                val = (
                    (2 * abs(nx - dimx / 2 + 0.5) / dimx) ** 2
                    + (2 * abs(nz - dimz / 2 + 0.5) / dimz) ** 2
                ) * (boxsize / radius)
                if val > 1:
                    x_true[nx][nz] = 0
                    x_est[nx][nz] = 0
                else:
                    npoints += 1
            #            if type_=='surface':
            if hist:
                x_t = x_true[nx][nz]
                x_e = x_est[nx][nz]
                if not np.isclose(x_t, 0, atol=treshold) and not np.isclose(
                    x_e, 0, atol=treshold
                ):
                    ae = abs(x_t - x_e)
                    rae = ae / x_t
                    histdata[nx, nz] = rae
    RMSE = np.sqrt(np.sum(np.multiply((x_true - x_est) ** 2, weightmatrix)) / npoints)
    MAE = np.sum(np.abs(np.multiply((x_est - x_true), weightmatrix))) / npoints
    MBE = np.sum(np.multiply((x_est - x_true), weightmatrix)) / npoints
    mean_true_abs = np.sum(np.multiply((x_true), weightmatrix)) / npoints
    AAD0 = np.sum(np.multiply(np.abs(x_true), weightmatrix)) / npoints
    AAD_est = np.sum(np.multiply(np.abs(x_est), weightmatrix)) / npoints

    NMAE = MAE / AAD0
    NMBE = MBE / AAD0
    NRMSE = RMSE / AAD0

    Data = [RMSE, MAE, MBE, NRMSE, NMAE, NMBE, AAD0, AAD_est]
    if AAD2 > 0:
        NMAE2 = MAE / AAD2
        NMBE2 = MBE / AAD2
        NRMSE2 = RMSE / AAD2
        Data.append(NRMSE2)
        Data.append(NMAE2)
        Data.append(NMBE2)
        Data.append(AAD2)

    # if hist:
    # Data.append(histdata)
    return Data


def full5tap3D(direction="x"):
    """
    parameters
    direction: either 'x', 'y' or 'z'
    """
    prefilter = -np.array([0.037659, 0.249153, 0.426375, 0.249153, 0.037659])
    prefilter1 = prefilter.reshape((5, 1))
    prefilter2 = prefilter.reshape((5, 1, 1))
    derivative0 = np.array([-0.104550, -0.292315, 0, 0.292315, 0.104550])
    if direction == "x":
        prefilterfull = np.kron(prefilter1, prefilter)
        derivative = derivative0.reshape((5, 1, 1))
    if direction == "y":
        prefilterfull = np.kron(prefilter, prefilter2)
        derivative = derivative0.reshape((5, 1))
    if direction == "z":
        prefilterfull = np.kron(prefilter1, prefilter2)
        derivative = derivative0
    full5tap = np.kron(prefilterfull, derivative)
    return full5tap


def full9tap3D(direction="x"):
    """
    parameters
    direction: either 'x', 'y' or 'z'
    """
    prefilter = -np.array(
        [
            0.000721,
            0.015486,
            0.090341,
            0.234494,
            0.317916,
            0.234494,
            0.090341,
            0.015486,
            0.000721,
        ]
    )
    prefilter1 = prefilter.reshape((9, 1))
    prefilter2 = prefilter.reshape((9, 1, 1))
    derivative0 = -np.array(
        [
            -0.003059,
            -0.035187,
            -0.118739,
            -0.143928,
            0,
            0.143928,
            0.118739,
            0.035187,
            0.003059,
        ]
    )
    if direction == "x":
        prefilterfull = np.kron(prefilter1, prefilter)
        derivative = derivative0.reshape((9, 1, 1))
    if direction == "y":
        prefilterfull = np.kron(prefilter, prefilter2)
        derivative = derivative0.reshape((9, 1))
    if direction == "z":
        prefilterfull = np.kron(prefilter1, prefilter2)
        derivative = derivative0
    full9tap = np.kron(prefilterfull, derivative)
    return full9tap


def set_axes_equal(ax, yscale=1):
    """Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def visSH3D_data3(
    cmesh=None, r0=1, lmax_plot=None, cmap="RdBu", vmin=None, vmax=None
):  # direction='r'):
    """Plot reconstructed spherical shape and traction colored 3d plot
    Returns
    -------
    fig,ax : matplotlib figure and axis instances
    Parameters
    ----------
    xmesh : ndarray, dimension (lmax+1, 2*lmax+1, nd)
        Mesh point representation of displacement SH vector
    cmesh : ndarray, dimension (lmax+1, 2*lmax+1, nd), optional
        If used, color the 3d shape with the mesh point representation
    r0 : float
        Radius of the original spherical shape
    lmax_plot : int, optional
        If used, the mesh is truncated to the given lmax;
        If None, determined by the mesh size
    """
    # cmesh1=np.swapaxes(cmesh0,0,-1)
    # cmesh=np.append(cmesh1,[cmesh1[:,:,0]],-1)
    if lmax_plot is None:
        lmax_plot = cmesh.shape[0] - 1
    lats, lons = _psh.expand.GLQGridCoord(lmax_plot)  # , extend=True)
    nlat = lats.size
    nlon = lons.size

    lats_circular = np.hstack(([90.0], lats, [-90.0]))
    lons_circular = np.append(lons, [lons[0]])
    u = np.radians(lons_circular)
    v = np.radians(90.0 - lats_circular)
    normvec = np.zeros((nlat + 2, nlon + 1, 3))
    """
    if direction=='theta':
        normvec[...,0] = np.cos(v)[:, None] * np.cos(u)[None, :]
        normvec[...,1] = np.cos(v)[:, None] * np.sin(u)[None, :]
        normvec[...,2] = -np.sin(v)[:, None] * np.ones_like(lons_circular)[None, :]
    elif direction=='r':
        normvec[...,0] = np.sin(v)[:, None] * np.cos(u)[None, :]
        normvec[...,1] = np.sin(v)[:, None] * np.sin(u)[None, :]
        normvec[...,2] = np.cos(v)[:, None] * np.ones_like(lons_circular)[None, :]
    elif direction=='phi':
        normvec[...,0] = -np.sin(u)[None, :]* np.ones_like(lons_circular)[None, :]
        normvec[...,1] = np.cos(u)[None, :]*  np.ones_like(lons_circular)[None, :]
        normvec[...,2] = np.cos(v)[:, None] * np.zeros_like(lons_circular)[None, :]
    elif direction=='x':
        normvec[...,0]=np.ones_like(normvec[...,0])
    elif direction=='y':
        normvec[...,1]=np.ones_like(normvec[...,1])
    elif direction=='z':
        normvec[...,2]=np.ones_like(normvec[...,2])
    """
    x = r0 * np.sin(v)[:, None] * np.cos(u)[None, :]
    y = r0 * np.sin(v)[:, None] * np.sin(u)[None, :]
    z = r0 * np.cos(v)[:, None] * np.ones_like(lons_circular)[None, :]
    tpoints = np.zeros((nlat + 2, nlon + 1, 3))
    # print(tpoints.shape)
    # print(cmesh.shape)
    cmesh = np.swapaxes(cmesh, 0, -1)
    cmesh = np.swapaxes(cmesh, 0, 1)
    tpoints[1:-1, :-1, 0] = cmesh
    tpoints[0, :, 0] = np.mean(cmesh[0, :], axis=0)  # not exact !
    tpoints[-1, :, 0] = np.mean(cmesh[-1, :], axis=0)  # not exact !
    tpoints[1:-1, -1, 0] = cmesh[:, 0]
    magn_point = np.sum(tpoints, axis=-1)
    # magn_point = np.sum(normvec*tpoints, axis=-1)
    # =plt.hist(magn_point.flatten())
    magn_face = (
        1.0
        / 4.0
        * (
            magn_point[1:, 1:]
            + magn_point[:-1, 1:]
            + magn_point[1:, :-1]
            + magn_point[:-1, :-1]
        )
    )
    magnmax_face = np.max(np.abs(magn_face))
    magnmax_point = np.max(np.abs(magn_point))
    if vmin is None:
        vmin = -magnmax_face
    if vmax is None:
        vmax = magnmax_face
    norm = mpl.colors.Normalize(vmin, vmax, clip=True)
    cmap = plt.get_cmap(cmap)
    colors = cmap(norm(magn_face.flatten()))
    colors = colors.reshape(nlat + 1, nlon, 4)
    return x, y, z, colors, cmap, norm  # ,magn_point


def extendflip(a):
    a2 = np.append(a, np.flip(a[:-1]))
    print(a.shape)
    print(a2.shape)
    return a2


def generate_submat(modes, mu, nu, lmax_col, lmax_row, shtype="irr", verbose=False):
    """Obtain the sub-matrix of spherical harmonic modes

    Parameters
    ----------
    modes : dict
        one of Umodes, Smodes, or Tmodes created by generate_modes
    mu,nu : float
        shear modulus and Poisson's ratio
    lmax_col,lmax_row : int
        lmax for column number (K), and row number (J)
    shtype : string, ['irr' or 'reg']
        'irr' represents irregular spherical harmonics for spherical void
        'reg' represents regular spherical harmonics for solid sphere
    verbose: bool
        if True, print sub-matrix size

    Returns
    -------
    complex lil_matrix, dimension (kJ*(lmax_row+1)^2, kK*(lmax_col+1)^2)
        sub-matrix of spherical harmonic mode. kK = 3 for 3 directions (i,j,k)
        kJ = 3 for vector (U and T), 9 for tensor (S)

    See Also
    --------
    generate_modes : generate spherical harmonic modes

    """
    if "U0" + shtype in modes.keys():  # obtain displacement mode full matrix
        U1 = modes["U1" + shtype]
        U0 = modes["U0" + shtype]
        fullmat = calUmode((U1, U0), mu, nu)
        kK, kJ = 3, 3
    elif "S0" + shtype in modes.keys():  # obtain stress mode full matrix
        S1 = modes["S1" + shtype]
        S2 = modes["S2" + shtype]
        S3 = modes["S3" + shtype]
        S0 = modes["S0" + shtype]
        fullmat = calSmode((S1, S2, S3, S0), mu, nu)
        kK, kJ = 3, 9
    elif "T0" + shtype in modes.keys():  # obtain traction mode full matrix
        T1 = modes["T1" + shtype]
        T2 = modes["T2" + shtype]
        T3 = modes["T3" + shtype]
        T0 = modes["T0" + shtype]
        fullmat = calSmode((T1, T2, T3, T0), mu, nu)
        kK, kJ = 3, 3
    else:
        print("input is not SH mode created by generate_modes()")
        return -1

    M, N = fullmat.shape
    lKfull = np.sqrt(N / kK).astype(int) - 1
    lJfull = np.sqrt(M / kJ).astype(int) - 1

    LJfull = (lJfull + 1) ** 2
    LKfull = (lKfull + 1) ** 2
    LJmax = (lmax_row + 1) ** 2
    LKmax = (lmax_col + 1) ** 2
    full_row = kJ * LJfull
    full_col = kK * LKfull
    size_row = kJ * LJmax
    size_col = kK * LKmax
    if verbose:
        print("Integrating modes to a matrix")
        print(size_row, size_col)

    # Combine sparse matrix using block matrices
    mat_blocks = [[None for _ in range(kK)] for _ in range(kJ)]
    for kj in range(kJ):
        for kk in range(kK):
            r1 = kj * LJmax
            r2 = r1 + LJmax
            c1 = kk * LKmax
            c2 = c1 + LKmax
            R1 = kj * LJfull
            R2 = R1 + LJmax
            C1 = kk * LKfull
            C2 = C1 + LKmax
            mat_blocks[kj][kk] = fullmat[R1:R2, C1:C2]
    submat = _spm.bmat(mat_blocks)

    return submat


def loadCoeffs(mu0, nu0, lmax, shtype, coeff_dir=None, verbose=True):
    mu = 1.0
    nu = nu0
    lJmax = lKmax = lmax
    if coeff_dir is None:
        coeff_dir = os.path.join("..", "shelastic", "default_modes")
    Dmat = generate_submat(
        loadmat(os.path.join(coeff_dir, "Umodes.mat")),
        mu,
        nu,
        lKmax,
        lJmax,
        shtype=shtype,
        verbose=verbose,
    ).tocsc()
    Cmat = generate_submat(
        loadmat(os.path.join(coeff_dir, "Tmodes.mat")),
        mu,
        nu,
        lKmax,
        lJmax,
        shtype=shtype,
        verbose=verbose,
    ).tocsc()
    return Cmat, Dmat
