# -*- coding: utf-8 -*-
"""
Ultimate RVE pipeline: STABLE FRACTURE VERSION
Optimized for Abaqus 2024 (Jython 2.7)
Usage: abaqus cae noGUI=Last.py
"""
import os, sys, math, random, csv
import numpy as np
from abaqus import *
from abaqusConstants import *
from caeModules import *
from odbAccess import openOdb

SIZE_X, SIZE_Y, SIZE_Z = 100.0, 100.0, 100.0
FIBER_RADIUS = 3.5
VF_TARGET = round(random.uniform(50.0, 70.0), 1)
AZIMUTH_VARIATION = 15.0
MAX_POLAR_ANGLE = 3.0
CSV_SHAPE = (100, 100, 100)
CSV_OUTPUT_DIR = 'rve_csv'

E_matrix = 2600.0
nu_matrix = 0.35
density_matrix = 1.24E-09
plastic_data = (
    (70.0, 0.0), (72.0, 0.05), (75.0, 0.1), (80.0, 0.2), (90.0, 0.4)
)
ductile_damage_data = ((0.05, 0.33, 1.0),)
damage_accumulation_power = 0.0
fracture_energy = 2.0

density_fiber = 1.8E-09
E_fiber_1 = 230000.0
E_fiber_2 = 15000.0
E_fiber_3 = 15000.0
nu_fiber_12 = 0.20
nu_fiber_13 = 0.20
nu_fiber_23 = 0.49
G_fiber_12 = 24000.0
G_fiber_13 = 24000.0
G_fiber_23 = 5030.0
fiber_tensile_strength_1 = 4900.0
fiber_compressive_strength_1 = 1400.0
fiber_transverse_strength = 50.0
fiber_fracture_energy = 0.12

MODEL_NAME = 'RVE_Model'
CAE_FILE = 'RVE_Model.cae'
JOB_NAME = 'RVE_Job'

STEP_TIME = 5.0  
TARGET_POINTS = 200
ODB_PATH = os.path.join(os.getcwd(), JOB_NAME + '.odb')
STRESS_CSV = os.path.join(os.getcwd(), 'strain_stress.csv')

def min_distance_between_segments(p1, p2, q1, q2):
    u = p2 - p1
    v = q2 - q1
    w = p1 - q1
    a = np.dot(u, u)
    b = np.dot(u, v)
    c = np.dot(v, v)
    d = np.dot(u, w)
    e = np.dot(v, w)
    D = a*c - b*b
    
    if D < 1e-12:
        s = 0.0
        t = e / c if c != 0 else 0.0
    else:
        s = (b*e - c*d) / D
        t = (a*e - b*d) / D

    if s < 0.0:
        s = 0.0
        t = e / c if c != 0 else 0.0
    elif s > 1.0:
        s = 1.0
        t = (b + e) / c if c != 0 else 0.0

    if t < 0.0:
        t = 0.0
        s = -d / a if a != 0 else 0.0
    elif t > 1.0:
        t = 1.0
        s = (b - d) / a if a != 0 else 0.0

    s = max(0.0, min(1.0, s))
    t = max(0.0, min(1.0, t))

    P = p1 + s * u
    Q = q1 + t * v
    return np.linalg.norm(P - Q)

class Fiber:
    __slots__ = ('center', 'radius', 'dir', 'length', 'half_len', 'p1', 'p2')
    
    def __init__(self, center, radius, polar_deg, azimuth_deg):
        self.center = np.array(center, dtype=float)
        self.radius = radius
        polar_rad = math.radians(polar_deg)
        azimuth_rad = math.radians(azimuth_deg)
        self.dir = np.array([math.sin(polar_rad)*math.cos(azimuth_rad),
                             math.sin(polar_rad)*math.sin(azimuth_rad),
                             math.cos(polar_rad)])
        self.length = SIZE_Z / max(1e-6, abs(self.dir[2]))
        self.half_len = self.length / 2.0
        self.p1 = self.center - self.half_len * self.dir
        self.p2 = self.center + self.half_len * self.dir

    def update_endpoints(self):
        self.p1 = self.center - self.half_len * self.dir
        self.p2 = self.center + self.half_len * self.dir

    def check_collision(self, other, tolerance=1e-4):
        center_dist = np.linalg.norm(self.center - other.center)
        max_dist = self.length + other.length + self.radius + other.radius
        if center_dist > max_dist:
            return False
        d = min_distance_between_segments(self.p1, self.p2, other.p1, other.p2)
        return d < (self.radius + other.radius - tolerance)

def resolve_collisions_fast(fibers, bounds, radius, max_iter=80, tol=1e-4):
    n = len(fibers)
    if n < 2: return
    xmin, xmax, ymin, ymax = bounds
    margin = radius
    cell_size = 2.2 * radius
    
    for iteration in range(max_iter):
        grid = {}
        for i, f in enumerate(fibers):
            gx = int((f.center[0] - xmin) / cell_size)
            gy = int((f.center[1] - ymin) / cell_size)
            if (gx, gy) not in grid:
                grid[(gx, gy)] = []
            grid[(gx, gy)].append(i)
        
        moved = False
        max_ov = 0.0
        
        for i in range(n):
            fi = fibers[i]
            gx = int((fi.center[0] - xmin) / cell_size)
            gy = int((fi.center[1] - ymin) / cell_size)
            
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    key = (gx + dx, gy + dy)
                    if key not in grid: continue
                    for j in grid[key]:
                        if j <= i: continue
                        fj = fibers[j]
                        
                        if abs(fi.center[0] - fj.center[0]) > 3.0*radius or \
                           abs(fi.center[1] - fj.center[1]) > 3.0*radius:
                            continue
                        
                        if fi.check_collision(fj, tolerance=tol):
                            d = min_distance_between_segments(fi.p1, fi.p2, fj.p1, fj.p2)
                            overlap = (fi.radius + fj.radius) - d
                            if overlap > 0:
                                max_ov = max(max_ov, overlap)
                                dx_c = fi.center[0] - fj.center[0]
                                dy_c = fi.center[1] - fj.center[1]
                                dist = math.hypot(dx_c, dy_c)
                                if dist < 1e-12:
                                    dx_c, dy_c, dist = 1.0, 0.0, 1.0
                                shift = overlap * 0.75
                                nx, ny = dx_c/dist, dy_c/dist
                                fi.center[0] += shift * nx
                                fi.center[1] += shift * ny
                                fj.center[0] -= shift * nx
                                fj.center[1] -= shift * ny
                                moved = True
        
        for f in fibers:
            f.center[0] = max(xmin + margin, min(xmax - margin, f.center[0]))
            f.center[1] = max(ymin + margin, min(ymax - margin, f.center[1]))
            f.update_endpoints()
        
        if not moved or max_ov < tol:
            break

class FiberGenerator:
    def __init__(self, radius, target_vf, max_pol, az_var):
        self.radius = radius
        self.target_vf = target_vf / 100.0
        self.max_pol = max_pol
        self.az_var = az_var
        self.spacing = 2 * radius * 1.005

    def _hex_positions(self, jitter=0.0):
        positions = []
        dx = self.spacing
        dy = self.spacing * math.sqrt(3) / 2.0
        nx = int(SIZE_X / dx) + 3
        ny = int(SIZE_Y / dy) +  3
        
        for i in range(nx):
            x = i * dx
            if x  < self.radius or x  > SIZE_X - self.radius: continue
            offset = 0 if (i % 2 == 0) else dx/2.0
            for j in range(ny):
                y = j * dy + offset
                if y  < self.radius or y  > SIZE_Y - self.radius: continue
                jx = x + random.uniform(-jitter, jitter)
                jy = y + random.uniform(-jitter, jitter)
                if self.radius  <= jx  <= SIZE_X - self.radius and \
                   self.radius  <= jy  <= SIZE_Y - self.radius:
                    positions.append((jx, jy))
        return positions

    def generate(self):
        n_req = max(1, int(round(self.target_vf * (SIZE_X*SIZE_Y*SIZE_Z) /  
                                 (math.pi * self.radius**2 * SIZE_Z))))
        print("Target fiber count: {}".format(n_req))
        
        pos = []
        for jt in [0.0, 0.3, 0.6, 1.0, 1.5]:
            pos = self._hex_positions(jitter=jt)
            if len(pos)  >= n_req:
                pos = random.sample(pos, n_req)
                break
        else:
            pos = self._hex_positions(jitter=2.0)
            if len(pos)  > n_req: pos = random.sample(pos, n_req)
        
        fibers = []
        base_az = random.uniform(0, 360)
        z_center = SIZE_Z / 2.0
        
        for cx, cy in pos:
            dist_to_edge = min(cx, SIZE_X - cx, cy, SIZE_Y - cy)
            margin = self.radius * 2.0
            if dist_to_edge  < margin:
                pol = 0.0
            else:
                fac = min(1.0, (dist_to_edge - margin) / (5 * self.radius))
                pol = self.max_pol * fac * random.uniform(0.8, 1.2 )
            az = (base_az + random.uniform(-self.az_var, self.az_var)) % 360.0
            fibers.append(Fiber([cx, cy, z_center], self.radius, pol, az))
        
        bounds = (0.0, SIZE_X,  0.0, SIZE_Y)
        print("Resolving collisions (spatial hash)... ")
        resolve_collisions_fast(fibers, bounds, self.radius)
        
        total_fib_vol = 0.0
        for f in fibers:
            total_fib_vol += math.pi * (f.radius ** 2) * f.length
        
        vol_rve = SIZE_X * SIZE_Y * SIZE_Z
        actual_vf = (total_fib_vol / vol_rve) * 100.0
        print("Generated {} non-intersecting fibers. Actual VF: {:.2f}% ".format(len(fibers), actual_vf))
        return fibers, actual_vf

def collect_metadata(actual_vf, density):
    return {
        'size_x_mm': SIZE_X, 'size_y_mm': SIZE_Y, 'size_z_mm': SIZE_Z,
        'raster_shape': CSV_SHAPE, 'fiber_radius_mm': FIBER_RADIUS,
        'actual_volume_fraction_%': actual_vf,
        'target_volume_fraction_%': VF_TARGET,
        'mean_raster_density': float(np.mean(density)),
        'min_raster_density': float(np.min(density)),
        'max_raster_density': float(np.max(density)),
        'matrix_E_MPa': E_matrix, 'matrix_nu': nu_matrix, 'matrix_density': density_matrix,
        'matrix_plastic_data': plastic_data,
        'matrix_ductile_damage_data': ductile_damage_data,
        'matrix_damage_accumulation_power': damage_accumulation_power,
        'matrix_fracture_energy_GJm2': fracture_energy,
        'fiber_density': density_fiber,
        'fiber_E1_MPa': E_fiber_1, 'fiber_E2_MPa': E_fiber_2, 'fiber_E3_MPa': E_fiber_3,
        'fiber_nu12': nu_fiber_12, 'fiber_nu13': nu_fiber_13, 'fiber_nu23': nu_fiber_23,
        'fiber_G12_MPa': G_fiber_12, 'fiber_G13_MPa': G_fiber_13, 'fiber_G23_MPa': G_fiber_23,
        'fiber_tensile_strength_MPa': fiber_tensile_strength_1,
        'fiber_compressive_strength_MPa': fiber_compressive_strength_1,
        'fiber_transverse_strength_MPa': fiber_transverse_strength,
        'fiber_fracture_energy_GJm2': fiber_fracture_energy,
    }

def rasterize_fibers_to_density_fast(fibers, shape, radius, size):
    nx, ny, nz = shape
    sx, sy, sz = size[0]/nx, size[1]/ny, size[2]/nz
    border_width = min(sx, sy, sz)
    x = np.arange(nx) * sx + sx/2.0
    y = np.arange(ny) * sy + sy/2.0
    z = np.arange(nz) * sz + sz/2.0
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    density = np.zeros((nx, ny, nz), dtype=np.float32)
    r2 = radius*radius
    bw2 = border_width*border_width

    for fib in fibers:
        c = fib.center
        d = fib.dir
        half_len = fib.half_len
        
        DX = X - c[0]
        DY = Y - c[1]
        DZ = Z - c[2]
        proj = DX*d[0] + DY*d[1] + DZ*d[2]
        t = np.clip(proj, -half_len, half_len)
        
        closest_x = c[0] + t * d[0]
        closest_y = c[1] + t * d[1]
        closest_z = c[2] + t * d[2]
        
        dist2 = (X - closest_x)**2 + (Y - closest_y)**2 + (Z - closest_z)**2
        
        val = np.zeros_like(dist2)
        inside = dist2 <= r2
        val[inside] = 1.0
        border = (dist2 > r2) & (dist2 <= r2 + bw2)
        if np.any(border):
            d_border = np.sqrt(dist2[border]) - radius
            val[border] = 1.0 - d_border / border_width
        density += val

    density = np.clip(density, 0.0, 1.0)
    return density

def save_csv_from_density(density, output_dir, metadata_dict):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    nz = density.shape[2]
    for z_idx in range(nz):
        layer = density[:, :, z_idx]
        csv_file = os.path.join(output_dir, 'rve_layer_{}.csv'.format(z_idx+1))
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            for row in layer:
                writer.writerow(["{:.6f}".format(v) for v in row])

    meta_file = os.path.join(output_dir, 'metadata.csv')
    with open(meta_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['parameter', 'value'])
        for k, v in metadata_dict.items():
            writer.writerow([k, str(v)])

    print("Density layers and full metadata saved to: {}".format(output_dir))
    print("Mean density = {:.6f}".format(metadata_dict['mean_raster_density']))

def create_model_and_inp(fibers):
    if MODEL_NAME in mdb.models: del mdb.models[MODEL_NAME]
    m = mdb.Model(name=MODEL_NAME)
    
    sk = m.ConstrainedSketch(name='MatSk', sheetSize=200)
    sk.rectangle((0,0), (SIZE_X, SIZE_Y))
    p_mat = m.Part(name='Matrix', dimensionality=THREE_D, type=DEFORMABLE_BODY)
    p_mat.BaseSolidExtrude(sketch=sk, depth=SIZE_Z)

    mat_m = m.Material('Matrix')
    mat_m.Density(table=((density_matrix,),))
    mat_m.Elastic(table=((E_matrix, nu_matrix),))
    mat_m.Plastic(table=plastic_data, hardening=ISOTROPIC)
    mat_m.DuctileDamageInitiation(table=ductile_damage_data, accumulationPower=damage_accumulation_power,
                                  rateDependency=OFF, temperatureDependency=OFF, dependencies=0)
    mat_m.ductileDamageInitiation.DamageEvolution(type=ENERGY, softening=LINEAR, degradation=MAXIMUM,
                                                   mixedModeBehavior=MODE_INDEPENDENT, modeMixRatio=ENERGY,
                                                  table=((fracture_energy,),), temperatureDependency=OFF, dependencies=0)
    m.HomogeneousSolidSection(name='SecMat', material='Matrix')
    p_mat.SectionAssignment(region=p_mat.Set(cells=p_mat.cells, name='All'), sectionName='SecMat')

    mat_f = m.Material('Fiber')
    mat_f.Density(table=((density_fiber,),))
    mat_f.Elastic(type=ENGINEERING_CONSTANTS,
                  table=((E_fiber_1 , E_fiber_2, E_fiber_3, nu_fiber_12, nu_fiber_13, nu_fiber_23,
                          G_fiber_12, G_fiber_13, G_fiber_23),))
    mat_f.MaxsDamageInitiation(
        table=((4900.0, 1400.0,  50.0, 150.0, 90.0, 50.0),),
        rateDependency=OFF, temperatureDependency=OFF, dependencies=0
    )
    mat_f.maxsDamageInitiation.DamageEvolution(type=ENERGY, softening=LINEAR, degradation=MAXIMUM,
                                               mixedModeBehavior=MODE_INDEPENDENT, modeMixRatio=ENERGY,
                                               table=((fiber_fracture_energy,),), temperatureDependency=OFF, dependencies=0)
    m.HomogeneousSolidSection(name='SecFib', material='Fiber')

    ass = m.rootAssembly
    ass.Instance(name='MatInst', part=p_mat, dependent=ON)

    el_type = mesh.ElemType(elemCode=C3D8R, elemLibrary=EXPLICIT, elemDeletion=ON, hourglassControl=ENHANCED)

    for i, f in enumerate(fibers):
        pn, sn = 'FibP_{}'.format(i), 'FibSk_{}'.format(i)
        fs = m.ConstrainedSketch(name=sn, sheetSize=10)
        fs.CircleByCenterPerimeter(center=(0,0), point1=(FIBER_RADIUS, 0))
        fp = m.Part(name=pn, dimensionality=THREE_D, type=DEFORMABLE_BODY)
        fp.BaseSolidExtrude(sketch=fs, depth=f.length)
        fp.SectionAssignment(region=fp.Set(cells=fp.cells, name='All'), sectionName='SecFib')
        
        csys_feature = fp.DatumCsysByDefault(CARTESIAN)
        fp.MaterialOrientation(
            region=fp.Set(cells=fp.cells, name='All'), 
            orientationType=SYSTEM, 
            localCsys=fp.datums[csys_feature.id], 
            axis=AXIS_3,
            additionalRotationType=ROTATION_ANGLE,
            angle=0.0
        )
        
        fp.seedPart(size=1.5, deviationFactor=0.1)
        fp.setElementType(regions=(fp.cells,), elemTypes=(el_type,))
        fp.generateMesh()
        
        iname = 'FibI_{}'.format(i)
        ass.Instance(name=iname, part=fp, dependent=ON)
        za = np.array([0,0,1])
        ra = np.cross(za, f.dir)
        nr = np.linalg.norm(ra)
        if nr  > 1e-12:
            ra /= nr
            ang = math.degrees(math.acos(np.clip(np.dot(za, f.dir), -1.0, 1.0)))
            ass.rotate(instanceList=(iname,), axisPoint=(0,0,0), axisDirection=tuple(ra), angle=ang)
        ass.translate(instanceList=(iname,), vector=(f.center[0], f.center[1], f.center[2] - f.length/2.0))

    for inst in ass.instances.values():
        if 'FibI_' in inst.name:
            ass.makeIndependent(instances=(inst,))
    ass.regenerate()

    host = ass.Set(cells=ass.instances['MatInst'].cells, name='HostRegion')
    for i in range(len(fibers)):
        inst = ass.instances['FibI_{}'.format(i)]
        fr = ass.Set(cells=inst.cells, name='FibSet_{}'.format(i))
        m.EmbeddedRegion(name='Emb_{}'.format(i), embeddedRegion=fr, hostRegion=host, absoluteTolerance=0.9)

    p_mat.seedPart(size=2.0, deviationFactor=0.1)
    p_mat.setElementType(regions=(p_mat.cells,), elemTypes=(el_type,))
    p_mat.generateMesh()

    mdb.saveAs(pathName=os.path.join(os.getcwd(), CAE_FILE))
    print("CAE saved: {}".format(CAE_FILE))

    print("Creating Reference Point at top face center...")
    rp_feature = ass.ReferencePoint(point=(SIZE_X/2.0, SIZE_Y/2.0, SIZE_Z))
    rp_obj = ass.referencePoints[rp_feature.id]
    rp_set = ass.Set(referencePoints=(rp_obj,), name='rp')

    top_faces = ass.instances['MatInst'].faces.findAt(((SIZE_X/2.0, SIZE_Y/2.0, SIZE_Z), ))
    ass.Surface(name='TopFaceSurface', side1Faces=top_faces)

    m.Coupling(name='rp_coupling', 
               controlPoint=ass.sets['rp'], 
               surface=ass.surfaces['TopFaceSurface'], 
               couplingType=KINEMATIC,
               influenceRadius=WHOLE_SURFACE,
                u1=ON, u2=ON, u3=ON, ur1=ON, ur2=ON, ur3=ON)

    bottom_faces = ass.instances['MatInst'].faces.findAt(((SIZE_X/2.0, SIZE_Y/2.0, 0.0), ))
    ass.Set(faces=bottom_faces, name='BottomFace')
    m.EncastreBC(name='BC-1_Stop', createStepName='Initial', region=ass.sets['BottomFace'])

    m.DisplacementBC(name='BC_RP', createStepName='Initial', region=ass.sets['rp'], 
                     u1=0, u2=0, u3=0, ur1=0, ur2=0, ur3=0)

    m.ExplicitDynamicsStep(
        name='Step-1', 
        previous='Initial', 
        description='Tension in Z-direction', 
        timePeriod=STEP_TIME, 
        nlgeom=ON, 
        adiabatic=OFF, 
        massScaling=((SEMI_AUTOMATIC, MODEL, AT_BEGINNING, 0, 1e-03, BELOW_MIN, 0, 0, 0, 0, 0, GLOBAL_NONE),)
    )

    m.SmoothStepAmplitude(name='AmpD', timeSpan=STEP, data=((0.0, 0.0), (STEP_TIME, 1.0)))
    
    target_displacement = 15.0 
    m.boundaryConditions['BC_RP'].setValuesInStep(stepName='Step-1', amplitude='AmpD', u3=target_displacement)

    field_output = m.fieldOutputRequests['F-Output-1']
    field_output.setValuesInStep(stepName='Step-1', 
                                 frequency=10, 
                                 variables=('S', 'SVAVG', 'PE', 'PEVAVG', 'PEEQ', 'PEEQVAVG', 'LE', 'U', 'V', 'A', 'RF', 'CSTRESS', 'EVF', 'STATUS', 'SDEG', 'SDV', 'ENER'))

    m.HistoryOutputRequest(
        name='H-Output-RP',
        createStepName='Step-1',
        region=ass.sets['rp'],
        variables=('U3', 'RF3'),
        numIntervals=500 
    )

    j = mdb.Job(name=JOB_NAME, model=MODEL_NAME, type=ANALYSIS)
    j.writeInput()
    print("Input file created: {}.inp".format(JOB_NAME))

def extract_results():
    if not os.path.exists(ODB_PATH):
        raise FileNotFoundError("ODB not found. Run the job first: abaqus job={} interactive".format(JOB_NAME))
    
    odb = openOdb(ODB_PATH, readOnly=True)
    try:
        step = odb.steps['Step-1']
        u3_data, rf3_data = None, None
        
        for hr_key in step.historyRegions.keys():
            hr = step.historyRegions[hr_key]
            if 'U3' in hr.historyOutputs and 'RF3' in hr.historyOutputs:
                u3_data = hr.historyOutputs['U3'].data
                rf3_data = hr.historyOutputs['RF3'].data
                print("History data loaded from region: '{}'".format(hr_key))
                break
        
        if u3_data is None or rf3_data is None:
            raise RuntimeError("History output with U3/RF3 not found in ODB.")
        
        u3 = np.array([v[1] for v in u3_data])
        rf3 = np.array([v[1] for v in rf3_data])
        
        strain = np.abs(u3) / SIZE_Z
        stress = np.abs(rf3) / (SIZE_X * SIZE_Y)

        max_stress_idx = np.argmax(stress)
        cutoff_idx = min(len(stress), max_stress_idx + int(0.05 * len(stress)))
        
        strain_cut = strain[:cutoff_idx]
        stress_cut = stress[:cutoff_idx]
        
        print("Peak Stress: {:.2f} MPa at Strain: {:.4f}".format(np.max(stress_cut), strain_cut[np.argmax(stress_cut)]))
        print("Data truncated from {} points to {} points (post-failure noise removed).".format(len(stress), len(stress_cut)))

        idx = np.linspace(0, len(strain_cut)-1, min(TARGET_POINTS, len(strain_cut))).astype(int)
        
        with open(STRESS_CSV, 'w', newline='') as cf:
            w = csv.writer(cf)
            w.writerow(['Strain', 'Stress_MPa'])
            w.writerows(zip(strain_cut[idx], stress_cut[idx]))
            
        print("Results exported to {}".format(STRESS_CSV))
        
    finally:
        odb.close()

def main():
    if not os.path.exists(ODB_PATH):
        print("MODE: Internal Generation & INP Creation")
        gen = FiberGenerator(FIBER_RADIUS, VF_TARGET, MAX_POLAR_ANGLE, AZIMUTH_VARIATION)
        fibers, actual_vf = gen.generate()
        if not fibers: raise RuntimeError("Fiber generation failed.")
        
        print("\nRasterizing to density grid...")
        density = rasterize_fibers_to_density_fast(fibers, shape=CSV_SHAPE, radius=FIBER_RADIUS, size=(SIZE_X, SIZE_Y, SIZE_Z))
        
        meta = collect_metadata(actual_vf, density)
        save_csv_from_density(density, CSV_OUTPUT_DIR, meta)
        
        print("\nBuilding Abaqus model and writing INP...")
        create_model_and_inp(fibers)
        
        print("\nNext step: Run job in terminal")
        print(">>> abaqus job={} interactive cpus=12".format(JOB_NAME))
    else:
        print("MODE: Post-Processing")
        try:
            extract_results()
            print("Done.")
        except Exception as e:
            print("Error during post-processing: {}".format(e))
            print("The job might have failed completely or ODB is corrupted.")

if __name__ == '__main__':
    main()
