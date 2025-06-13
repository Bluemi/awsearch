import numpy as np


def rgb2xyz(rgb):
    """Convert an (N,3) array of sRGB [0–1] to CIE XYZ."""
    rgb = rgb.copy()
    mask = rgb > 0.04045
    rgb[mask] = ((rgb[mask] + 0.055) / 1.055) ** 2.4
    rgb[~mask] /= 12.92
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    return rgb @ M.T


def xyz2lab(xyz):
    """Convert an (N,3) array of XYZ to CIELAB."""
    wp = np.array([0.95047, 1.00000, 1.08883])
    xyz_scaled = xyz / wp
    mask = xyz_scaled > 0.008856
    f = np.where(mask, xyz_scaled ** (1/3), (7.787 * xyz_scaled + 16/116))
    L = 116 * f[:,1] - 16
    a = 500 * (f[:,0] - f[:,1])
    b = 200 * (f[:,1] - f[:,2])
    return np.stack([L, a, b], axis=1)


def lab2xyz(lab):
    """Convert an (N,3) array of CIELAB to XYZ."""
    L, a, b = lab[:,0], lab[:,1], lab[:,2]
    fy = (L + 16) / 116
    fx = a / 500 + fy
    fz = fy - b / 200
    fx3, fz3 = fx**3, fz**3
    xyz = np.zeros_like(lab)
    xyz[:,0] = np.where(fx3 > 0.008856, fx3, (fx - 16/116) / 7.787)
    xyz[:,1] = np.where(L > (903.3 * 0.008856), fy**3, L / 903.3)
    xyz[:,2] = np.where(fz3 > 0.008856, fz3, (fz - 16/116) / 7.787)
    wp = np.array([0.95047, 1.00000, 1.08883])
    return xyz * wp


def xyz2rgb(xyz):
    """Convert an (N,3) array of XYZ to sRGB [0–1]."""
    M = np.array([[ 3.2404542, -1.5371385, -0.4985314],
                  [-0.9692660,  1.8760108,  0.0415560],
                  [ 0.0556434, -0.2040259,  1.0572252]])
    rgb = xyz @ M.T
    mask = rgb > 0.0031308
    rgb[mask] = 1.055 * (rgb[mask] ** (1/2.4)) - 0.055
    rgb[~mask] *= 12.92
    return np.clip(rgb, 0, 1)


def ensure_high_contrast(colors, threshold=20, max_iter=1000) -> np.ndarray:
    """
    colors: (N,3) uint8 RGB
    returns: (N,3) uint8 RGB with ΔE ≥ threshold
    """
    rgb = colors.astype(float) / 255.0
    lab = xyz2lab(rgb2xyz(rgb))
    for _ in range(max_iter):
        d = np.linalg.norm(lab[:,None,:] - lab[None,:,:], axis=-1)
        close = (d < threshold) & (~np.eye(len(lab),dtype=bool))
        if not np.any(close):
            break
        i, j = np.argwhere(close)[0]
        lab[i] += np.random.normal(scale=threshold/2, size=3)
    rgb_out = xyz2rgb(lab2xyz(lab))
    return (rgb_out * 255).astype(np.uint8)
