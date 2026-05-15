# 在仓库根执行：部分环境下 `python scripts/generate_demo_nc_three_modules.py` 写 NC 会 PermissionError，
# 本脚本用 python -c 内联生成（与手动调试行为一致）。
Set-Location $PSScriptRoot\..
$py = @'
import json, sys
from pathlib import Path
import numpy as np
from netCDF4 import Dataset

ROOT = Path('.').resolve()
# 勿用绝对路径传给 netCDF4（Windows + 中文仓库路径会 PermissionError）
OUT = Path('outputs') / 'demo_nc_three_modules'
OUT.mkdir(parents=True, exist_ok=True)

def dims(nc, nt, lat, lon):
    nc.createDimension('time', nt)
    nc.createDimension('lat', len(lat))
    nc.createDimension('lon', len(lon))
    nc.createVariable('time', 'i4', ('time',))[:] = np.arange(nt, dtype=np.int32)
    nc.createVariable('lat', 'f4', ('lat',))[:] = lat.astype(np.float32)
    nc.createVariable('lon', 'f4', ('lon',))[:] = lon.astype(np.float32)

meta = {'generated_by': 'scripts/run_generate_demo_nc.ps1', 'modules': {}}

# mod1
p1 = OUT / 'mod1_ocean_sst_uv.nc'
if p1.exists(): p1.unlink()
rng = np.random.default_rng(42)
nt, ny, nx = 2, 64, 64
lat = np.linspace(31., 41., ny, dtype=np.float32)
lon = np.linspace(117., 127., nx, dtype=np.float32)
yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float32)
sst = np.empty((nt, ny, nx), np.float32)
ssu = np.empty((nt, ny, nx), np.float32)
ssv = np.empty((nt, ny, nx), np.float32)
for it in range(nt):
    base = 22.0 + 0.4 * np.sin(xx * 0.15) * np.cos(yy * 0.12) + 0.15 * it
    sst[it] = base + rng.standard_normal((ny, nx)).astype(np.float32) * 0.3
    ssu[it] = 0.05 * np.sin(yy * 0.1 + it) + rng.standard_normal((ny, nx)).astype(np.float32) * 0.02
    ssv[it] = 0.05 * np.cos(xx * 0.1 - it) + rng.standard_normal((ny, nx)).astype(np.float32) * 0.02
nc = Dataset(str(p1.as_posix()), 'w', format='NETCDF4')
nc.setncatts({'title': 'mod1 SST SSU SSV', 'module_hint': 'vortex'})
dims(nc, nt, lat, lon)
for name, arr in [('SST', sst), ('SSU', ssu), ('SSV', ssv)]:
    v = nc.createVariable(name, 'f4', ('time', 'lat', 'lon'), zlib=True, complevel=4)
    v[:] = arr
nc.close()
meta['modules']['vortex'] = {'file': str(p1.resolve().relative_to(ROOT)), 'vars': ['SST', 'SSU', 'SSV']}

# mod2
p2 = OUT / 'mod2_ocean_wind_wave.nc'
if p2.exists(): p2.unlink()
rng = np.random.default_rng(43)
nt, ny, nx = 96, 12, 12
lat = np.linspace(30., 40., ny, dtype=np.float32)
lon = np.linspace(118., 126., nx, dtype=np.float32)
u10 = rng.normal(6., 1.5, (nt, ny, nx)).astype(np.float32)
v10 = rng.normal(0., 2., (nt, ny, nx)).astype(np.float32)
spd = np.sqrt(np.maximum(u10, 0) ** 2 + v10 ** 2)
swh = (0.08 * spd + 0.3 * rng.random((nt, ny, nx)).astype(np.float32)).astype(np.float32)
nc = Dataset(str(p2.as_posix()), 'w', format='NETCDF4')
nc.setncatts({'title': 'mod2 u10 v10 swh', 'module_hint': 'wind_wave'})
dims(nc, nt, lat, lon)
for name, arr in [('u10', u10), ('v10', v10), ('swh', swh)]:
    v = nc.createVariable(name, 'f4', ('time', 'lat', 'lon'), zlib=True, complevel=4)
    v[:] = arr
nc.close()
meta['modules']['wind_wave'] = {'file': str(p2.resolve().relative_to(ROOT)), 'vars': ['u10', 'v10', 'swh']}

# mod3
p3 = OUT / 'mod3_ocean_hydro_L2.nc'
if p3.exists(): p3.unlink()
rng = np.random.default_rng(44)
nt, ny, nx = 240, 138, 125
lat = np.linspace(31., 41., ny, dtype=np.float32)
lon = np.linspace(117., 127., nx, dtype=np.float32)
t1 = np.linspace(0, 2 * np.pi, nt, dtype=np.float32)[:, None, None]
yy = np.linspace(0, 1, ny, dtype=np.float32)[None, :, None]
xx = np.linspace(0, 1, nx, dtype=np.float32)[None, None, :]
sst = (20.0 + 0.8 * np.sin(t1) + 0.3 * np.sin(4 * xx) * np.cos(3 * yy)).astype(np.float32)
sst = sst + rng.standard_normal((nt, ny, nx)).astype(np.float32) * 0.05
sss = (34.5 + 0.1 * np.cos(t1 * 0.5) + rng.standard_normal((nt, ny, nx)).astype(np.float32) * 0.02).astype(np.float32)
ssu = (0.15 * np.sin(t1 + yy) + rng.standard_normal((nt, ny, nx)).astype(np.float32) * 0.02).astype(np.float32)
ssv = (0.15 * np.cos(t1 - xx) + rng.standard_normal((nt, ny, nx)).astype(np.float32) * 0.02).astype(np.float32)
nc = Dataset(str(p3.as_posix()), 'w', format='NETCDF4')
nc.setncatts({'title': 'mod3 hydro L2 T=240', 'module_hint': 'hydro'})
dims(nc, nt, lat, lon)
for name, arr in [('SST', sst), ('sss', sss), ('SSU', ssu), ('SSV', ssv)]:
    v = nc.createVariable(name, 'f4', ('time', 'lat', 'lon'), zlib=True, complevel=4)
    v[:] = arr
nc.close()
meta['modules']['hydro'] = {'file': str(p3.resolve().relative_to(ROOT)), 'vars': ['SST', 'sss', 'SSU', 'SSV'], 'T': nt}

(OUT / 'demo_nc_manifest.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(meta, ensure_ascii=False, indent=2))
'@
python -c $py
