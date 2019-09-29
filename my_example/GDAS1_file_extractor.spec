# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['GDAS1_file_extractor.py'],
             pathex=['C:\\Users\\zhenping\\Documents\\Python Scripts\\MyLib\\ARLreader\\my_example'],
             binaries=[],
             datas=[],
             hiddenimports=['numpy'],
             hookspath=['C:\\Users\\zhenping\\Documents\\Python Scripts\\MyLib\\ARLreader'],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='GDAS1_file_extractor',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True )
