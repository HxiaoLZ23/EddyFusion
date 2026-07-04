"""在 Windows / 中文路径下更稳健地用 xarray 打开 NetCDF4(HDF5) 数据集。"""

from __future__ import annotations

import shutil
import tempfile
import zlib
from pathlib import Path
from typing import Any


def open_xr_dataset_compat(nc_path: Path | str) -> tuple[Any, Path | None]:
    """打开命题方等大体积 NetCDF（底层常为 HDF）。

    Windows 下若仓库路径含中文，部分 ``netCDF4+/libhdf`` 会以 ``FileNotFoundError``
    报告无法打开路径，此时 ``open(..., 'rb')`` 仍可读。依次尝试：

    1. ``h5netcdf``（若已安装）；
    2. xarray 默认引擎；
    3. ``copy`` 至 ``%TEMP%`` 下纯 ASCII 文件名再打开。

    返回 ``(Dataset, 临时文件路径或 None)``——调用方须在 ``dataset.close()`` 后删除临时文件。
    """

    import xarray as xr

    nc = Path(nc_path).resolve()

    trials: list[dict[str, str]] = []
    try:
        import h5netcdf  # noqa: F401, PLC0415

        trials.append({"engine": "h5netcdf"})
    except ImportError:
        pass
    trials.append({})

    last_exc: BaseException | None = None
    for kw in trials:
        try:
            return xr.open_dataset(nc, **kw), None
        except ImportError:
            # 未安装 h5py 等依赖时跳过该引擎
            continue
        except (FileNotFoundError, OSError) as e:
            last_exc = e
            continue

    if last_exc is None:
        raise RuntimeError(
            "无法打开 NetCDF：xarray 所有可用引擎均失败（请确认已安装 netCDF4 / h5netcdf+h5py）"
        )

    key = zlib.crc32(str(nc).encode("utf-8")) & 0xFFFFFFFF
    dup = Path(tempfile.gettempdir()) / f"eddy_nc_compat_{key}_{nc.stem}.nc"
    print(
        f"[xarray_nc_open] xarray 从原路径打开失败 ({type(last_exc).__name__})，使用临时拷贝: {dup}",
        flush=True,
    )
    shutil.copy2(nc, dup)
    try:
        return xr.open_dataset(dup), dup
    except BaseException:
        dup.unlink(missing_ok=True)
        raise
