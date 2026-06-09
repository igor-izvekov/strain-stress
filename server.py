# -*- coding: utf-8 -*-
"""
FastAPI сервер: RVE генерация → 3D CSV → CNN-GPR инференс
Запуск: python main.py
Открыть: http://127.0.0.1:8000/
Требования: файл index.html должен лежать в той же папке
"""
import os, io, uuid, math, random, csv, json
import numpy as np
from PIL import Image
from fastapi import FastAPI, Form, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

try:
    import tensorflow as tf
    from tensorflow import keras
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.preprocessing import StandardScaler
    import joblib
    from scipy.interpolate import interp1d
    from scipy import ndimage
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

PORT = 8000
TEMP_DIR = "./temp_results"
MODEL_DIR = "./cnn_gpr"
os.makedirs(TEMP_DIR, exist_ok=True)

SIZE_X, SIZE_Y, SIZE_Z = 100.0, 100.0, 100.0
FIBER_RADIUS = 3.5
CSV_SHAPE = (100, 100, 100)
N_STRAIN_POINTS = 200

E_matrix, nu_matrix = 2600.0, 0.35
density_matrix = 1.24E-09
plastic_data = [(70.0, 0.0), (72.0, 0.05), (75.0, 0.1), (80.0, 0.2), (90.0, 0.4)]
fracture_energy_matrix = 2.0

density_fiber = 1.8E-09
E_fiber_1 = 230000.0
E_fiber_2 = E_fiber_3 = 15000.0
nu_fiber_12 = nu_fiber_13 = 0.20
nu_fiber_23 = 0.49
G_fiber_12 = G_fiber_13 = 24000.0
G_fiber_23 = 5030.0
fiber_tensile_strength_1 = 4900.0
fiber_fracture_energy = 0.12

app = FastAPI(title="RVE → CNN-GPR Predictor")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

hybrid_model = None

def load_hybrid_model(model_dir: str):
    if not ML_AVAILABLE:
        return None
    try:
        files_needed = ["cnn_extractor.keras", "gpr_heads.pkl", "scaler_features.pkl", "scaler_target.pkl"]
        if not all(os.path.exists(os.path.join(model_dir, f)) for f in files_needed):
            raise FileNotFoundError("Не все файлы модели найдены")
        
        cnn = keras.models.load_model(os.path.join(model_dir, "cnn_extractor.keras"), compile=False)
        with open(os.path.join(model_dir, "gpr_heads.pkl"), "rb") as f:
            gpr_heads = joblib.load(f)
        with open(os.path.join(model_dir, "scaler_features.pkl"), "rb") as f:
            scaler_feat = joblib.load(f)
        with open(os.path.join(model_dir, "scaler_target.pkl"), "rb") as f:
            scaler_tgt = joblib.load(f)
        
        return {"cnn": cnn, "gpr_heads": gpr_heads, "scaler_feat": scaler_feat, "scaler_tgt": scaler_tgt}
    except Exception as e:
        return None

if ML_AVAILABLE and os.path.exists(MODEL_DIR):
    hybrid_model = load_hybrid_model(MODEL_DIR)

def min_distance_between_segments(p1, p2, q1, q2):
    u, v, w = p2 - p1, q2 - q1, p1 - q1
    a, b, c = np.dot(u,u), np.dot(u,v), np.dot(v,v)
    d, e = np.dot(u,w), np.dot(v,w)
    D = a*c - b*b
    if D < 1e-12:
        s, t = 0.0, (e/c if c != 0 else 0.0)
    else:
        s, t = (b*e - c*d)/D, (a*e - b*d)/D
    s, t = np.clip(s, 0, 1), np.clip(t, 0, 1)
    return np.linalg.norm((p1 + s*u) - (q1 + t*v))

class Fiber:
    __slots__ = ('center', 'radius', 'dir', 'length', 'half_len', 'p1', 'p2')
    def __init__(self, center, radius, polar_deg, azimuth_deg):
        self.center = np.array(center, dtype=float)
        self.radius = radius
        pr, az = math.radians(polar_deg), math.radians(azimuth_deg)
        self.dir = np.array([math.sin(pr)*math.cos(az), math.sin(pr)*math.sin(az), math.cos(pr)])
        self.length = SIZE_Z / max(1e-6, abs(self.dir[2]))
        self.half_len = self.length / 2.0
        self.p1 = self.center - self.half_len * self.dir
        self.p2 = self.center + self.half_len * self.dir
        
    @classmethod
    def from_endpoints(cls, p1, p2, radius):
        """Создание волокна по двум конечным точкам (для соединения срезов)."""
        p1 = np.array(p1, dtype=float)
        p2 = np.array(p2, dtype=float)
        center = (p1 + p2) / 2.0
        direction = p2 - p1
        length = np.linalg.norm(direction)
        if length < 1e-6:
            dir_norm = np.array([0.0, 0.0, 1.0])
        else:
            dir_norm = direction / length
            
        obj = cls.__new__(cls)
        obj.center = center
        obj.radius = radius
        obj.dir = dir_norm
        obj.length = length
        obj.half_len = length / 2.0
        obj.p1 = p1
        obj.p2 = p2
        return obj

    def update_endpoints(self):
        self.p1 = self.center - self.half_len * self.dir
        self.p2 = self.center + self.half_len * self.dir
        
    def check_collision(self, other, tol=1e-4):
        if np.linalg.norm(self.center - other.center) > self.length + other.length + 2*self.radius:
            return False
        return min_distance_between_segments(self.p1, self.p2, other.p1, other.p2) < (self.radius + other.radius - tol)

def resolve_collisions_fast(fibers, bounds, radius, max_iter=80, tol=1e-4):
    n = len(fibers)
    if n < 2: return
    xmin, xmax, ymin, ymax = bounds
    margin, cell_size = radius, 2.2 * radius
    for _ in range(max_iter):
        grid = {}
        for i, f in enumerate(fibers):
            key = (int((f.center[0]-xmin)/cell_size), int((f.center[1]-ymin)/cell_size))
            grid.setdefault(key, []).append(i)
        moved, max_ov = False, 0.0
        for i in range(n):
            fi = fibers[i]
            gx, gy = int((fi.center[0]-xmin)/cell_size), int((fi.center[1]-ymin)/cell_size)
            for dx in (-1,0,1):
                for dy in (-1,0,1):
                    for j in grid.get((gx+dx, gy+dy), []):
                        if j <= i: continue
                        fj = fibers[j]
                        if abs(fi.center[0]-fj.center[0])>3*radius or abs(fi.center[1]-fj.center[1])>3*radius: continue
                        if fi.check_collision(fj, tol):
                            d = min_distance_between_segments(fi.p1, fi.p2, fj.p1, fj.p2)
                            overlap = (fi.radius + fj.radius) - d
                            if overlap > 0:
                                max_ov = max(max_ov, overlap)
                                dx_c, dy_c = fi.center[0]-fj.center[0], fi.center[1]-fj.center[1]
                                dist = math.hypot(dx_c, dy_c) + 1e-12
                                shift = overlap * 0.75
                                fi.center[0] += shift * dx_c/dist
                                fi.center[1] += shift * dy_c/dist
                                fj.center[0] -= shift * dx_c/dist
                                fj.center[1] -= shift * dy_c/dist
                                moved = True
        for f in fibers:
            f.center[0] = np.clip(f.center[0], xmin+margin, xmax-margin)
            f.center[1] = np.clip(f.center[1], ymin+margin, ymax-margin)
            f.update_endpoints()
        if not moved or max_ov < tol: break

class FiberGenerator:
    def __init__(self, radius, target_vf, max_pol, az_var):
        self.radius, self.target_vf = radius, target_vf/100.0
        self.max_pol, self.az_var = max_pol, az_var
        self.spacing = 2 * radius * 1.005
        
    def _hex_positions(self, jitter=0.0):
        pos, dx = [], self.spacing
        dy = dx * math.sqrt(3) / 2.0
        for i in range(int(SIZE_X/dx)+3):
            x = i * dx
            if x < self.radius or x > SIZE_X-self.radius: continue
            offset = 0 if i%2==0 else dx/2.0
            for j in range(int(SIZE_Y/dy)+3):
                y = j * dy + offset
                if y < self.radius or y > SIZE_Y-self.radius: continue
                jx, jy = x+random.uniform(-jitter,jitter), y+random.uniform(-jitter,jitter)
                if self.radius <= jx <= SIZE_X-self.radius and self.radius <= jy <= SIZE_Y-self.radius:
                    pos.append((jx, jy))
        return pos
        
    def generate(self):
        n_req = max(1, int(round(self.target_vf * SIZE_X*SIZE_Y / (math.pi * self.radius**2))))
        pos = []
        for jt in [0.0, 0.3, 0.6, 1.0, 1.5]:
            pos = self._hex_positions(jitter=jt)
            if len(pos) >= n_req:
                pos = random.sample(pos, n_req)
                break
        else:
            pos = self._hex_positions(jitter=2.0)
            if len(pos) > n_req: pos = random.sample(pos, n_req)
            
        fibers, base_az = [], random.uniform(0, 360)
        for cx, cy in pos:
            dist = min(cx, SIZE_X-cx, cy, SIZE_Y-cy)
            pol = 0.0 if dist < 2*self.radius else self.max_pol * min(1.0, (dist-2*self.radius)/(5*self.radius)) * random.uniform(0.8,1.2)
            az = (base_az + random.uniform(-self.az_var, self.az_var)) % 360
            fibers.append(Fiber([cx, cy, SIZE_Z/2], self.radius, pol, az))
            
        resolve_collisions_fast(fibers, (0,SIZE_X,0,SIZE_Y), self.radius)
        total_vol = sum(math.pi * f.radius**2 * f.length for f in fibers)
        actual_vf = (total_vol / (SIZE_X*SIZE_Y*SIZE_Z)) * 100
        print(f"Generated {len(fibers)} fibers. Actual VF: {actual_vf:.2f}%")
        return fibers, actual_vf

def rasterize_fibers_to_density_fast(fibers, shape, radius, size):
    nx, ny, nz = shape
    sx, sy, sz = size[0]/nx, size[1]/ny, size[2]/nz
    bw = min(sx, sy, sz)
    x = np.arange(nx)*sx + sx/2
    y = np.arange(ny)*sy + sy/2
    z = np.arange(nz)*sz + sz/2
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    density = np.zeros((nx,ny,nz), dtype=np.float32)
    r2, bw2 = radius**2, bw**2
    for fib in fibers:
        c, d, hl = fib.center, fib.dir, fib.half_len
        DX, DY, DZ = X-c[0], Y-c[1], Z-c[2]
        t = np.clip(DX*d[0]+DY*d[1]+DZ*d[2], -hl, hl)
        cx, cy, cz = c[0]+t*d[0], c[1]+t*d[1], c[2]+t*d[2]
        dist2 = (X-cx)**2 + (Y-cy)**2 + (Z-cz)**2
        val = np.zeros_like(dist2)
        inside = dist2 <= r2
        val[inside] = 1.0
        border = (dist2 > r2) & (dist2 <= r2 + bw2)
        if np.any(border):
            val[border] = 1.0 - (np.sqrt(dist2[border]) - radius) / bw
        density += val
    return np.clip(density, 0.0, 1.0)

def find_fiber_centers(image_path):
    img = Image.open(image_path).convert('L')
    arr = np.array(img)
    
    hist, _ = np.histogram(arr, bins=256, range=(0, 256))
    total = arr.size
    sum_total = np.sum(np.arange(256) * hist)
    
    sum_bg, weight_bg, max_variance, best_threshold = 0, 0, 0, 128
    for i in range(256):
        weight_bg += hist[i]
        if weight_bg == 0: continue
        weight_fg = total - weight_bg
        if weight_fg == 0: break
        sum_bg += i * hist[i]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if variance > max_variance:
            max_variance = variance
            best_threshold = i
            
    binary_light = arr > best_threshold
    binary_dark = arr <= best_threshold
    binary = binary_light if np.sum(binary_light) < np.sum(binary_dark) else binary_dark
    
    binary = ndimage.binary_opening(binary, structure=np.ones((2, 2)))
    
    labeled, num_features = ndimage.label(binary)
    centers = []
    h, w = arr.shape
    min_area, max_area = 5, h * w * 0.2
    
    binary_arr = binary.astype(int)
    for i in range(1, num_features + 1):
        area = np.sum(labeled == i)
        if min_area < area < max_area:
            cy, cx = ndimage.center_of_mass(binary_arr, labeled, i)
            phys_x = (cx / w) * SIZE_X
            phys_y = (cy / h) * SIZE_Y
            centers.append((phys_x, phys_y))
            
    return centers

def match_centers(centers_a, centers_b):
    """Жадное сопоставление центров между двумя срезами по минимальному расстоянию."""
    dists = []
    for i, (xa, ya) in enumerate(centers_a):
        for j, (xb, yb) in enumerate(centers_b):
            dist = math.hypot(xa - xb, ya - yb)
            dists.append((dist, i, j))
    dists.sort()
    
    pairs, used_a, used_b = [], set(), set()
    for dist, i, j in dists:
        if i not in used_a and j not in used_b:
            pairs.append((centers_a[i], centers_b[j]))
            used_a.add(i)
            used_b.add(j)
    return pairs

def run_inference(density_3d: np.ndarray, vf_target: float, n_samples: int = None):
    if not ML_AVAILABLE or hybrid_model is None:
        return predict_properties_fallback(vf_target)
    try:
        m = hybrid_model
        X_rve = density_3d[np.newaxis, ..., np.newaxis]
        X_meta = np.array([[vf_target]])
        latent = m["cnn"].predict(X_rve, verbose=0)
        X_comb = np.hstack([latent, X_meta])
        X_scaled = m["scaler_feat"].transform(X_comb)
        
        mean_curve, std_curve = [], []
        for gpr in m["gpr_heads"]:
            mu, std = gpr.predict(X_scaled, return_std=True)
            mean_curve.append(mu[0])
            std_curve.append(std[0])
        
        mean_curve = np.array(mean_curve).reshape(1,-1)
        std_curve = np.array(std_curve).reshape(1,-1)
        mean_curve = m["scaler_tgt"].inverse_transform(mean_curve)[0]
        std_curve = std_curve[0] * m["scaler_tgt"].scale_
        
        strain = np.linspace(0, 0.15, N_STRAIN_POINTS)
        
        max_idx = np.argmax(mean_curve)
        mean_max = mean_curve[max_idx]
        std_at_max = std_curve[max_idx]
        
        if n_samples is not None and n_samples < 100:
            k_A = 2.5 + 10 / np.sqrt(n_samples)
            k_B = 1.4 + 5 / np.sqrt(n_samples)
        else:
            k_A = 2.326
            k_B = 1.282
            
        a_basis = max(0, mean_max - k_A * std_at_max)
        b_basis = max(0, mean_max - k_B * std_at_max)
        
        return {
            "strain": [float(f"{s:.6f}") for s in strain],
            "stress": [float(f"{t:.2f}") for t in mean_curve],
            "stress_std": [float(f"{s:.2f}") for s in std_curve],
            "a_basis": round(float(a_basis), 2),
            "b_basis": round(float(b_basis), 2),
            "max_strength": round(float(mean_max), 2),
            "vf_used": vf_target,
            "uncertainty": True,
            "k_factors": {"k_A": round(k_A, 3), "k_B": round(k_B, 3)},
            "statistics": {
                "mean_max": round(float(mean_max), 2),
                "std_at_max": round(float(std_at_max), 2)
            }
        }
    except Exception as e:
        return predict_properties_fallback(vf_target)

def predict_properties_fallback(vf_target: float):
    vf = vf_target / 100.0
    E_comp = vf * E_fiber_1 + (1-vf) * E_matrix
    sigma_ult = (vf * fiber_tensile_strength_1 * 0.85 + (1-vf) * 90.0 * 0.3)
    strain_yield = 90.0 / E_matrix * (1 + vf * 0.4)
    strain_peak = 0.018 + vf * 0.007
    
    strain = np.linspace(0, 0.15, 500)
    stress = np.zeros_like(strain)
    
    for i, eps in enumerate(strain):
        if eps <= strain_yield:
            stress[i] = E_comp * eps
        elif eps <= strain_peak:
            plastic = eps - strain_yield
            hardening = 90.0 * (1 + 150.0 * (plastic ** 0.18))
            stress[i] = hardening * (E_comp / E_matrix)
        else:
            damage = 1 - np.exp(-(eps - strain_peak) / 0.025)
            stress[i] = sigma_ult * (1 - damage * 0.85) + sigma_ult * 0.15 * damage
            
    stress = np.maximum(stress + np.random.normal(0, sigma_ult*0.015, len(strain)), 0)
    stress = np.convolve(stress, np.ones(7)/7, mode='same')
    
    peak_idx = np.argmax(stress)
    cutoff = min(len(stress), peak_idx + int(0.05 * len(strain)))
    strain, stress = strain[:cutoff], stress[:cutoff]
    
    if len(strain) < N_STRAIN_POINTS:
        f = interp1d(strain, stress, kind='cubic', fill_value='extrapolate', bounds_error=False)
        strain_new = np.linspace(0, 0.15, N_STRAIN_POINTS)
        stress = f(strain_new)
        strain = strain_new
        
    max_s = float(np.max(stress))
    return {
        "strain": [float(f"{s:.6f}") for s in strain],
        "stress": [float(f"{t:.2f}") for t in stress],
        "stress_std": [float(f"{t*0.04:.2f}") for t in stress],
        "a_basis": round(max_s * 0.84, 2),
        "b_basis": round(max_s * 0.91, 2),
        "max_strength": round(max_s, 2),
        "vf_used": vf_target,
        "uncertainty": False,
        "note": "Fallback: физическая модель"
    }

@app.get("/", response_class=HTMLResponse)
async def root():
    index_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    if not os.path.exists(index_path):
        return HTMLResponse(content="<h1>❌ index.html не найден</h1><p>Положите файл index.html в папку с main.py</p>", status_code=404)
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.post("/predict/")
async def predict(vf_target: float = Form(60.0)):
    """Эндпоинт 1: Генерация RVE и инференс по заданному объему волокна (Vf)."""
    job_id = str(uuid.uuid4())
    job_dir = os.path.join(TEMP_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)
    try:
        if not (50 <= vf_target <= 70):
            raise HTTPException(400, "Vf должно быть 50–70%")
            
        gen = FiberGenerator(FIBER_RADIUS, vf_target, max_pol=3.0, az_var=15.0)
        fibers, actual_vf = gen.generate()
        if not fibers:
            raise HTTPException(500, "Не удалось сгенерировать волокна")
            
        density = rasterize_fibers_to_density_fast(fibers, CSV_SHAPE, FIBER_RADIUS, (SIZE_X, SIZE_Y, SIZE_Z))
        
        csv_dir = os.path.join(job_dir, "rve_csv")
        os.makedirs(csv_dir, exist_ok=True)
        for z in range(CSV_SHAPE[2]):
            with open(os.path.join(csv_dir, f"rve_layer_{z+1}.csv"), "w", newline='') as f:
                writer = csv.writer(f)
                for row in density[:, :, z]:
                    writer.writerow([f"{v:.6f}" for v in row])
        with open(os.path.join(csv_dir, "metadata.json"), "w") as f:
            json.dump({"actual_vf_%": round(actual_vf,2), "fiber_count": len(fibers), "shape": list(CSV_SHAPE)}, f, indent=2)
            
        result = run_inference(density, vf_target)
        
        curve_path = os.path.join(job_dir, "curve.csv")
        with open(curve_path, "w", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Strain", "Stress_MPa"])
            for s, t in zip(result["strain"], result["stress"]):
                writer.writerow([f"{s:.6f}", f"{t:.2f}"])
                
        return JSONResponse(content={"curve_url": f"/file/{job_id}/curve.csv", "csv_dir_url": f"/file/{job_id}/rve_csv/", **result})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Внутренняя ошибка: {str(e)}")

@app.post("/predict/from_slices/")
async def predict_from_slices(slice_a: UploadFile = File(...), slice_b: UploadFile = File(...)):
    """Эндпоинт 2: Генерация RVE по двум микроснимкам (начальный и конечный срезы)."""
    job_id = str(uuid.uuid4())
    job_dir = os.path.join(TEMP_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)
    
    try:
        path_a = os.path.join(job_dir, "slice_a.png")
        path_b = os.path.join(job_dir, "slice_b.png")
        with open(path_a, "wb") as f:
            f.write(await slice_a.read())
        with open(path_b, "wb") as f:
            f.write(await slice_b.read())
            
        centers_a = find_fiber_centers(path_a)
        centers_b = find_fiber_centers(path_b)
        
        if not centers_a or not centers_b:
            raise HTTPException(400, "Не удалось найти центры волокон на одном или обоих срезах")
            
        pairs = match_centers(centers_a, centers_b)
        if not pairs:
            raise HTTPException(400, "Не удалось сопоставить центры волокон между срезами")
            
        fibers = []
        for (xa, ya), (xb, yb) in pairs:
            p1 = np.array([xa, ya, 0.0])
            p2 = np.array([xb, yb, SIZE_Z])
            fibers.append(Fiber.from_endpoints(p1, p2, FIBER_RADIUS))
            
        total_vol = sum(math.pi * f.radius**2 * f.length for f in fibers)
        actual_vf = (total_vol / (SIZE_X * SIZE_Y * SIZE_Z)) * 100
        
        density = rasterize_fibers_to_density_fast(fibers, CSV_SHAPE, FIBER_RADIUS, (SIZE_X, SIZE_Y, SIZE_Z))
        
        csv_dir = os.path.join(job_dir, "rve_csv")
        os.makedirs(csv_dir, exist_ok=True)
        for z in range(CSV_SHAPE[2]):
            with open(os.path.join(csv_dir, f"rve_layer_{z+1}.csv"), "w", newline='') as f:
                writer = csv.writer(f)
                for row in density[:, :, z]:
                    writer.writerow([f"{v:.6f}" for v in row])
        with open(os.path.join(csv_dir, "metadata.json"), "w") as f:
            json.dump({"actual_vf_%": round(actual_vf, 2), "fiber_count": len(fibers), "shape": list(CSV_SHAPE)}, f, indent=2)
            
        result = run_inference(density, actual_vf)
        
        curve_path = os.path.join(job_dir, "curve.csv")
        with open(curve_path, "w", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Strain", "Stress_MPa"])
            for s, t in zip(result["strain"], result["stress"]):
                writer.writerow([f"{s:.6f}", f"{t:.2f}"])
                
        return JSONResponse(content={
            "curve_url": f"/file/{job_id}/curve.csv", 
            "csv_dir_url": f"/file/{job_id}/rve_csv/", 
            **result
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Внутренняя ошибка: {str(e)}")

@app.get("/file/{job_id}/{filename:path}")
async def serve_file(job_id: str, filename: str):
    path = os.path.join(TEMP_DIR, job_id, filename)
    if not os.path.exists(path):
        raise HTTPException(404, "Файл не найден")
    if os.path.isdir(path):
        return JSONResponse({"files": [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]})
    return FileResponse(path)

if __name__ == "__main__":
    print(f"Сервер: http://127.0.0.1:{PORT}")
    print(f"Модель: {MODEL_DIR} | ML: {ML_AVAILABLE}")
    print(f"Интерфейс: {os.path.join(os.getcwd(), 'index.html')}")
    uvicorn.run(app, host="127.0.0.1", port=PORT)