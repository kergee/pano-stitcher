# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
import os

from PyInstaller.utils.hooks import collect_submodules

root = Path(SPECPATH).parent

hiddenimports = [
    "app.main",
    "app.stitcher",
    "scipy.optimize",
]
excludes = []

if os.environ.get("PANO_BUNDLE_DEEP") == "1":
    for optional in ("kornia", "torch"):
        try:
            hiddenimports.extend(collect_submodules(optional))
        except Exception:
            pass
else:
    excludes.extend(["kornia", "kornia_rs", "torch"])

a = Analysis(
    [str(root / "tools" / "pano_server.py")],
    pathex=[str(root)],
    binaries=[],
    datas=[(str(root / "app" / "static"), "app/static")],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="pano-backend",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
