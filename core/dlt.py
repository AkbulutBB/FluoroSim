"""
core/dlt.py  —  X-ray projection via the normalised DLT
=======================================================

Each stored X-ray (AP and lateral) is modelled as a pin-hole projection of the
model-space world onto the image.  We recover its 3x4 projection matrix P from
>= 6 fiducial correspondences (steel bearings whose 3-D model coordinates you
know, clicked in the X-ray).  Then any model-space point — including the live
probe tip — can be projected onto that X-ray with  p ~ P @ [X Y Z 1]^T.

The DLT here is the *normalised* (Hartley) variant: points are isotropically
rescaled before the SVD and the result is denormalised afterwards.  This keeps
the design matrix well conditioned (pixel magnitudes are hundreds; model
magnitudes are tens of mm) and gives a lower, more trustworthy reprojection
error.
"""

from __future__ import annotations
import numpy as np


def _normalize_2d(pts: np.ndarray):
    c = pts.mean(axis=0)
    d = np.sqrt(((pts - c) ** 2).sum(axis=1)).mean()
    s = np.sqrt(2.0) / d if d > 1e-12 else 1.0
    T = np.array([[s, 0, -s * c[0]],
                  [0, s, -s * c[1]],
                  [0, 0,        1.0]])
    ph = np.hstack([pts, np.ones((len(pts), 1))])
    return (T @ ph.T).T[:, :2], T


def _normalize_3d(pts: np.ndarray):
    c = pts.mean(axis=0)
    d = np.sqrt(((pts - c) ** 2).sum(axis=1)).mean()
    s = np.sqrt(3.0) / d if d > 1e-12 else 1.0
    U = np.array([[s, 0, 0, -s * c[0]],
                  [0, s, 0, -s * c[1]],
                  [0, 0, s, -s * c[2]],
                  [0, 0, 0,        1.0]])
    ph = np.hstack([pts, np.ones((len(pts), 1))])
    return (U @ ph.T).T[:, :3], U


class ProjectionMatrix:
    """A 3x4 model->pixel projection for one X-ray view."""

    __slots__ = ("P",)

    def __init__(self, P: np.ndarray):
        self.P = np.asarray(P, dtype=np.float64).reshape(3, 4)

    @classmethod
    def from_correspondences(cls, obj_pts: np.ndarray, img_pts: np.ndarray) -> "ProjectionMatrix":
        obj = np.asarray(obj_pts, dtype=np.float64)
        img = np.asarray(img_pts, dtype=np.float64)
        n = len(obj)
        if n < 6:
            raise ValueError(f"DLT needs at least 6 fiducials (got {n}).")
        if len(img) != n:
            raise ValueError("3-D and 2-D fiducial counts differ.")

        Xn, U = _normalize_3d(obj)
        xn, T = _normalize_2d(img)

        A = []
        for (X, Y, Z), (u, v) in zip(Xn, xn):
            A.append([-X, -Y, -Z, -1,  0,  0,  0,  0, u * X, u * Y, u * Z, u])
            A.append([ 0,  0,  0,  0, -X, -Y, -Z, -1, v * X, v * Y, v * Z, v])
        A = np.asarray(A, dtype=np.float64)

        _, _, Vt = np.linalg.svd(A)
        P_n = Vt[-1].reshape(3, 4)
        P = np.linalg.inv(T) @ P_n @ U
        nrm = np.linalg.norm(P[2, :3])
        if not np.isfinite(nrm) or nrm < 1e-12:
            raise ValueError("Degenerate fiducial geometry - the coordinates must be "
                             "distinct and not all collinear/coplanar in a way the X-ray "
                             "cannot resolve. (Did every bearing get the same coordinate?)")
        P /= nrm                         # consistent scale for a finite camera
        return cls(P)

    def project(self, pt_model: np.ndarray) -> np.ndarray:
        """Project one model point (3,) or many (N,3) to pixel coordinates."""
        pt = np.asarray(pt_model, dtype=np.float64)
        single = pt.ndim == 1
        pts = pt.reshape(1, 3) if single else pt
        ph = np.hstack([pts, np.ones((len(pts), 1))])
        uvw = (self.P @ ph.T).T
        px = uvw[:, :2] / uvw[:, 2:3]
        return px[0] if single else px

    def reprojection_error(self, obj_pts: np.ndarray, img_pts: np.ndarray) -> float:
        """Mean Euclidean pixel error of the fiducials (your registration quality gauge)."""
        proj = self.project(np.asarray(obj_pts, dtype=np.float64))
        return float(np.mean(np.linalg.norm(proj - np.asarray(img_pts, dtype=np.float64), axis=1)))

    def tolist(self):
        return self.P.tolist()

    @classmethod
    def fromlist(cls, data):
        return cls(np.asarray(data, dtype=np.float64))
