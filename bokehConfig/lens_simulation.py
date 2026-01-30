"""
Biotar Lens Simulation Module
Python implementation matching the HLSL lens simulation (CameraLensData.hlsli, LensSimulation.hlsli)
"""
import numpy as np
from PIL import Image

# === CONFIGURATION (matches HLSL CameraLensData.hlsli) ===
n_air, v_air = 1.0, 89.30
focal_length = 50.0  # mm

# === APERTURE TEXTURE ===
_aperture_texture = None
_aperture_texture_path = None

def setApertureTexturePath(path):
    """Set the path to the aperture texture image."""
    global _aperture_texture_path, _aperture_texture
    _aperture_texture_path = path
    _aperture_texture = None  # Reset cache

def loadApertureTexture():
    global _aperture_texture
    if _aperture_texture is None:
        if _aperture_texture_path is None:
            raise ValueError("Aperture texture path not set. Call setApertureTexturePath() first.")
        _aperture_texture = np.asarray(Image.open(_aperture_texture_path).convert("L"), dtype=np.float32) / 255.0
    return _aperture_texture

def sampleHeliosApertureMask(uv):
    """Sample aperture mask at UV [0,1]. Black=open, white=blocked (HLSL convention)."""
    tex = loadApertureTexture()
    h, w = tex.shape
    x, y = int(round(uv[0] * (w - 1))), int(round(uv[1] * (h - 1)))
    return 1.0 - tex[max(0, min(h-1, y)), max(0, min(w-1, x))]

# === LENS ELEMENT CLASS (matches HLSL LensElement struct) ===
class LensElement:
    """Lens element matching HLSL struct with camelCase property names."""
    def __init__(self, curvatureRadius, thickness, n, v, apertureRadius, applyOpeningTexture=False):
        self.curvatureRadius = curvatureRadius
        self.thickness = thickness
        self.n, self.v = n, v
        self.apertureRadius = apertureRadius
        self.applyOpeningTexture = applyOpeningTexture

def createLensElement(curvatureRadius, thickness, n, v, apertureRadius, applyOpeningTexture=False):
    """Factory function matching HLSL createLensElement."""
    return LensElement(curvatureRadius, thickness, n, v, apertureRadius, applyOpeningTexture)

# === BIOTAR LENS DATA (matches HLSL CameraLensData.hlsli) ===
_r_p = np.array([83.6, 321.0, 44.8, -1150.0, 28.3, -38.5, 50.5, -53.2, 106.0, -120.0])
_d_p = np.array([10.75, 15.55, 5.05, 5.05, 21.22, 13.9])
_l_p = np.array([1.65, 18.9, 0.97])
_n = np.array([1.64238, 1.62306, 1.57566, 1.67270, 1.64238, 1.64238])
_v = np.array([48.0, 56.9, 41.2, 32.2, 48.0, 48.0])
_measured_aperture = np.array([3.0, 8.0, 12.0, 15.0, 17.2, 19.0, 20.0])
_a_px = np.array([455, 406, 270, 362])
_d_px = np.array([71, 100, 30, 30, 134, 86])
_l2_split = 73 / (73 + 46)

# Convert to mm (matches patent_to_mm, pixel_to_patent in HLSL)
_pmm = focal_length / 100.0
_pxp = np.mean(_d_p / _d_px)
r_curv = _r_p * _pmm
a = _a_px * _pxp * _pmm * 0.5
d = _d_p * _pmm
l = _l_p * _pmm

lens_element_count = 11  # matches HLSL lens_element_count

def createLensElements(aperture_stop_index, d_to_film):
    """Create lens elements matching HLSL lens_elements array."""
    ap_r = _measured_aperture[aperture_stop_index] * (a[2] / (_measured_aperture[6] * 0.5)) * 0.5
    return [
        createLensElement(r_curv[0], d[0],              _n[0], _v[0], a[1]),
        createLensElement(r_curv[1], l[0],              n_air, v_air, a[1]),
        createLensElement(r_curv[2], d[1],              _n[1], _v[1], a[1]),
        createLensElement(r_curv[3], d[2],              _n[2], _v[2], a[1]),
        createLensElement(r_curv[4], l[1] * _l2_split,  n_air, v_air, a[2]),
        createLensElement(0.0, l[1] * (1 - _l2_split),  n_air, v_air, ap_r, True),  # Aperture stop
        createLensElement(r_curv[5], d[3],              _n[3], _v[3], a[2]),
        createLensElement(r_curv[6], d[4],              _n[4], _v[4], a[3]),
        createLensElement(r_curv[7], l[2],              n_air, v_air, a[3]),
        createLensElement(r_curv[8], d[5],              _n[5], _v[5], a[3]),
        createLensElement(r_curv[9], d_to_film,         n_air, v_air, a[3]),
    ]

# === RAY TRACING CORE (matches HLSL LensSimulation.hlsli) ===

def sphereRayIntersect(rayOrigin, rayDir, radius):
    """Ray-sphere intersection for sphere at origin. Matches HLSL sphereRayIntersect."""
    a = np.dot(rayDir, rayDir)
    if a <= 1e-12: return False, 0.0, 0.0
    b, c = np.dot(rayOrigin, rayDir), np.dot(rayOrigin, rayOrigin) - radius * radius
    disc = b * b - a * c
    if disc < 0: return False, 0.0, 0.0
    sqrt_d = np.sqrt(disc)
    t0, t1 = (-b - sqrt_d) / a, (-b + sqrt_d) / a
    return True, min(t0, t1), max(t0, t1)

def intersect(radius, center, rayOrigin, rayDirection):
    """Intersect ray with lens surface. Matches HLSL intersect function."""
    shifted = rayOrigin - np.array([0.0, 0.0, center])
    success, t0, t1 = sphereRayIntersect(shifted, rayDirection, abs(radius))
    if not success: return False, 0.0, np.zeros(3)
    useCloserT = (rayDirection[2] > 0) ^ (radius < 0)
    t = t0 if useCloserT else t1
    hit = rayOrigin + t * rayDirection
    normal = (hit - np.array([0.0, 0.0, center]))
    normal = normal / np.linalg.norm(normal)
    if not useCloserT: normal = -normal
    return True, t, normal

def getEtaForWavelength(n_D, v_D, wavelength):
    """Wavelength-dependent refractive index via Cauchy's equation. Matches HLSL getEtaForWavelength."""
    lambda_F, lambda_D, lambda_C = 486.1327, 589.2938, 656.2725
    B = ((n_D - 1.0) / v_D) / ((1.0 / lambda_F**2) - (1.0 / lambda_C**2))
    A = n_D - B / lambda_D**2
    return A + B / wavelength**2

def refract(direction, normal, eta):
    """Refract direction. Returns None on total internal reflection. Matches HLSL refract."""
    cos_i = -np.dot(normal, direction)
    sin2_t = eta * eta * (1.0 - cos_i * cos_i)
    if sin2_t > 1.0: return None
    return eta * direction + (eta * cos_i - np.sqrt(max(0, 1 - sin2_t))) * normal

def traceLensesFromFilm(ray_origin, ray_direction, wavelength, elements):
    """Trace ray from film through lens toward scene. Matches HLSL traceLensesFromFilm."""
    origin, direction = np.array(ray_origin, float), np.array(ray_direction, float)
    direction = direction / np.linalg.norm(direction)
    path, z = [origin.copy()], 0.0
    
    for i in range(len(elements) - 1, -1, -1):
        elem = elements[i]
        n_I = elem.n if wavelength == 0 else getEtaForWavelength(elem.n, elem.v, wavelength)
        n_T = (elements[i-1].n if i > 0 else n_air) if wavelength == 0 else \
              getEtaForWavelength(elements[i-1].n if i > 0 else n_air, elements[i-1].v if i > 0 else v_air, wavelength)
        z -= elem.thickness
        
        if abs(elem.curvatureRadius) < 1e-9:
            if direction[2] >= 0: return False, None, None, np.array(path)
            t = (z - origin[2]) / direction[2]
            hit, normal = origin + t * direction, np.array([0.0, 0.0, 1.0])
        else:
            success, t, normal = intersect(elem.curvatureRadius, z + elem.curvatureRadius, origin, direction)
            if not success: return False, None, None, np.array(path)
            hit = origin + t * direction
        
        if elem.applyOpeningTexture:
            uv = hit[:2] / elem.apertureRadius * 0.5 + 0.5
            if not (0 <= uv[0] <= 1 and 0 <= uv[1] <= 1) or sampleHeliosApertureMask(uv) < 0.5:
                path.append(hit.copy()); return False, None, None, np.array(path)
        elif hit[0]**2 + hit[1]**2 > elem.apertureRadius**2:
            path.append(hit.copy()); return False, None, None, np.array(path)
        
        origin = hit.copy()
        path.append(origin.copy())
        
        if abs(elem.curvatureRadius) >= 1e-9:
            eta = n_I / (n_T if n_T > 0 else 1.0)
            refracted = refract(direction, normal, eta)
            if refracted is None: return False, None, None, np.array(path)
            direction = refracted / np.linalg.norm(refracted)
    
    return True, origin, direction, np.array(path)

def intersectSurfaceForward(origin, direction, curvature, z_surface):
    """Intersect ray with lens surface (forward direction, from scene toward film)."""
    if abs(curvature) < 1e-9:
        if abs(direction[2]) < 1e-12: return None
        t = (z_surface - origin[2]) / direction[2]
        if t <= 0.0: return None
        return origin + t * direction, np.array([0.0, 0.0, -np.sign(direction[2])])
    
    radius, center = curvature, z_surface + curvature
    shifted = origin - np.array([0.0, 0.0, center])
    a, b, c = np.dot(direction, direction), np.dot(shifted, direction), np.dot(shifted, shifted) - abs(radius)**2
    if a <= 1e-12: return None
    disc = b*b - a*c
    if disc < 0.0: return None
    sqrt_disc = np.sqrt(disc)
    t0, t1 = (-b - sqrt_disc) / a, (-b + sqrt_disc) / a
    if t0 > t1: t0, t1 = t1, t0
    useCloserT = (direction[2] > 0) ^ (radius < 0)
    t = t0 if useCloserT else t1
    if t <= 0.0: return None
    hit = origin + t * direction
    normal = hit - np.array([0.0, 0.0, center])
    norm_len = np.linalg.norm(normal)
    normal = np.array([0.0, 0.0, -np.sign(curvature)]) if norm_len < 1e-9 else normal / norm_len
    if not useCloserT: normal = -normal
    return hit, normal

def traceRayForward(origin, target, lensElements, recordPath=False, ignoreAperture=False):
    """Trace ray from scene through lens toward film (forward direction)."""
    origin, direction = np.array(origin, dtype=float), np.array(target, dtype=float) - np.array(origin, dtype=float)
    norm_dir = np.linalg.norm(direction)
    if norm_dir == 0.0: return None
    direction /= norm_dir
    path = [origin.copy()] if recordPath else None
    z_surf, n_curr = 0.0, n_air
    
    for elem in lensElements:
        result = intersectSurfaceForward(origin, direction, elem.curvatureRadius, z_surf)
        if result is None: return None
        hit, normal = result
        if hit[0]**2 + hit[1]**2 > elem.apertureRadius**2: return None
        if not ignoreAperture and elem.applyOpeningTexture:
            uv = hit[:2] / elem.apertureRadius * 0.5 + 0.5
            if not (0 <= uv[0] <= 1 and 0 <= uv[1] <= 1) or sampleHeliosApertureMask(uv) < 0.5: return None
        refracted = refract(direction, normal, n_curr / elem.n)
        if refracted is None: return None
        direction, origin = refracted / np.linalg.norm(refracted), hit
        if recordPath: path.append(origin.copy())
        n_curr, z_surf = elem.n, z_surf + elem.thickness
    return {"position": origin, "direction": direction, "path": np.vstack(path) if recordPath else None}

# === LENS VISUALIZATION ===

def getMaxR(curvature, aperture):
    """Max r for spherical surface (can't extend beyond sphere radius)."""
    return min(aperture, abs(curvature)) if abs(curvature) > 1e-9 else aperture

def sphericalProfile(curvature, z_surface, aperture, samples=240, force_max_r=None):
    """Generate profile points for lens surface."""
    if aperture <= 0: return None
    max_r = force_max_r if force_max_r else getMaxR(curvature, aperture)
    r = np.linspace(-max_r, max_r, samples)
    if abs(curvature) < 1e-9:
        return np.full_like(r, z_surface), r
    radius = abs(curvature)
    z_center = z_surface + curvature
    return z_center - np.sign(curvature) * np.sqrt(radius**2 - np.clip(r, -radius, radius)**2), r

def computeLensSurfaceZ(elements):
    """Compute z-position of each lens surface. Film at z=0, lens extends into negative z."""
    z_positions, z = [], 0.0
    for i in range(len(elements) - 1, -1, -1):
        z -= elements[i].thickness
        z_positions.append(z)
    return z_positions[::-1]

def intersectSurfaceBackward(origin, direction, curvature, z_surface):
    """Intersect ray with lens surface (backward direction, from film toward scene)."""
    if abs(curvature) < 1e-9:
        if abs(direction[2]) < 1e-12:
            return None
        t = (z_surface - origin[2]) / direction[2]
        if t <= 0.0:
            return None
        hit = origin + t * direction
        normal = np.array([0.0, 0.0, -1.0 if direction[2] > 0 else 1.0])
        return hit, normal
    
    radius = abs(curvature)
    center_z = z_surface + curvature
    oc = origin - np.array([0.0, 0.0, center_z])
    a, b, c = np.dot(direction, direction), np.dot(oc, direction), np.dot(oc, oc) - radius * radius
    if a <= 1e-12:
        return None
    disc = b * b - a * c
    if disc < 0.0:
        return None
    
    sqrt_disc = np.sqrt(disc)
    t0, t1 = (-b - sqrt_disc) / a, (-b + sqrt_disc) / a
    use_closer = (direction[2] > 0.0) ^ (curvature < 0.0)
    t = min(t0, t1) if use_closer else max(t0, t1)
    if t <= 0.0:
        return None
    
    hit = origin + t * direction
    normal = (hit - np.array([0.0, 0.0, center_z])) / radius
    return hit, normal if use_closer else -normal

def traceRayBackward(origin, target, elements, lensSurfaceZ=None, apertureThreshold=0.5):
    """
    Trace ray from film through lens toward scene (backward direction).
    Matches HLSL traceLensesFromFilm but with explicit path tracking.
    
    Parameters:
        origin: Starting point on/near film plane
        target: Target point to compute direction
        elements: List of LensElement objects
        lensSurfaceZ: Pre-computed z-positions (or None to compute)
        apertureThreshold: Threshold for aperture texture (default 0.5)
    
    Returns:
        (result_dict, path) where result_dict has 'position' and 'direction', or (None, path) if blocked
    """
    if lensSurfaceZ is None:
        lensSurfaceZ = computeLensSurfaceZ(elements)
    
    origin, direction = np.array(origin, dtype=float), np.array(target, dtype=float) - np.array(origin, dtype=float)
    path = [origin.copy()]
    norm = np.linalg.norm(direction)
    if norm == 0:
        return None, path
    direction /= norm
    
    n_current = n_air  # Start in air (after last element)
    for idx in range(len(elements) - 1, -1, -1):
        elem = elements[idx]
        z_surf = lensSurfaceZ[idx]
        
        result = intersectSurfaceBackward(origin, direction, elem.curvatureRadius, z_surf)
        if result is None:
            return None, path
        hit, normal = result
        path.append(hit.copy())
        
        # Check aperture
        x, y = hit[0], hit[1]
        if x * x + y * y > elem.apertureRadius ** 2:
            return None, path
        if elem.applyOpeningTexture:
            uv = np.array([0.5 + 0.5 * x / elem.apertureRadius, 0.5 + 0.5 * y / elem.apertureRadius])
            if not (0.0 <= uv[0] <= 1.0 and 0.0 <= uv[1] <= 1.0):
                return None, path
            if sampleHeliosApertureMask(uv) < apertureThreshold:
                return None, path
        
        # Get refractive index of next medium (toward scene)
        n_target = elements[idx - 1].n if idx > 0 else n_air
        if n_target <= 0:
            return None, path
        
        refracted = refract(direction, normal, n_current / n_target)
        if refracted is None:
            return None, path
        direction = refracted / np.linalg.norm(refracted)
        origin, n_current = hit, n_target
    
    return {"position": origin, "direction": direction}, path

def addLensProfile(ax, elements, glassColor="#71a7dd", glassAlpha=0.35):
    """Add lens profile to axes. Film at z=0, lens extends into negative z."""
    z_positions = computeLensSurfaceZ(elements)
    
    surfaces = []
    for idx, elem in enumerate(elements):
        z_surf = z_positions[idx]
        profile = sphericalProfile(elem.curvatureRadius, z_surf, elem.apertureRadius)
        surfaces.append((elem.curvatureRadius, z_surf, elem.apertureRadius, elem.n))
        if profile: ax.plot(profile[0], profile[1], color="black", linewidth=1.1)
    
    for idx in range(len(surfaces) - 1):
        curv_a, z_a, ap_a, n = surfaces[idx]
        curv_b, z_b, ap_b, _ = surfaces[idx + 1]
        if n <= n_air + 1e-6: continue
        fill_r = min(getMaxR(curv_a, ap_a), getMaxR(curv_b, ap_b))
        if fill_r <= 0: continue
        pa = sphericalProfile(curv_a, z_a, ap_a, 240, fill_r)
        pb = sphericalProfile(curv_b, z_b, ap_b, 240, fill_r)
        if pa and pb:
            ax.fill_betweenx(pa[1], np.minimum(pa[0], pb[0]), np.maximum(pa[0], pb[0]), 
                            color=glassColor, alpha=glassAlpha, linewidth=0)
