"""Targeted counterexamples from the research report; run via unittest discover."""
import ast
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from app.stitcher import autocrop, localwarp
from tools.edge_experiment import extend_valid_colors
from app.stitcher.budget import ComposeBudget, canvas_roi, estimate_bytes, plan_projection
from app.stitcher.features import ImageFeats, match_record
from app.stitcher.imageio import read_image, write_image, save_manifest, manifest_paths
from app.stitcher.pipeline import _Pair, _components, _max_spanning_tree, _bundle_adjust, stitch_pro, StitchError


class Files(unittest.TestCase):
    def test_unicode_long_and_spaces(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            for _ in range(12):path=path/'中文 长路径abc'
            path/='照片.png';image=np.full((8,9,3),127,np.uint8)
            path.parent.mkdir(parents=True,exist_ok=True)
            saved=path.with_name(path.name+'.tmp');saved.write_bytes(b'user temporary file')
            write_image(path,image);np.testing.assert_array_equal(read_image(path),image)
            self.assertEqual(saved.read_bytes(),b'user temporary file')

    def test_bad_decode(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'坏.jpg';path.write_bytes(b'broken')
            with self.assertRaises((ValueError,cv2.error)):read_image(path)

    def test_failed_write_preserves_existing(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'out.png';path.write_bytes(b'original')
            with patch.object(Path,'replace',side_effect=OSError('denied')):
                with self.assertRaises(OSError):write_image(path,np.zeros((8,8,3),np.uint8))
            self.assertEqual(path.read_bytes(),b'original');self.assertFalse(path.with_name('out.png.tmp').exists())

    def test_manifest_excludes_output_and_detects_changed_input(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);im=np.zeros((8,8,3),np.uint8)
            paths=[folder/'0.png',folder/'1.png']
            for p in paths:write_image(p,im)
            manifest=save_manifest(folder,paths);write_image(folder/'old-pano.png',im)
            self.assertEqual(len(manifest_paths(manifest)),2)
            write_image(paths[0],im+1)
            with self.assertRaises(ValueError):manifest_paths(manifest)

    def test_cli_failure_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            for i in range(2):write_image(folder/f'{i}.png',np.zeros((16,16,3),np.uint8))
            result=subprocess.run([sys.executable,'-m','app.stitcher.cli',str(folder),'--backend','sift'],capture_output=True)
            self.assertNotEqual(result.returncode,0)


class Budgets(unittest.TestCase):
    def test_canvas_not_roi_sum(self):
        rois=[(0,0,100,100),(100000,100000,100,100)]
        self.assertEqual(canvas_roi(rois)[2:],(100100,100100))
        self.assertGreater(estimate_bytes(rois),600_000_000_000)

    def test_invalid_roi(self):
        for value in [float("nan"),float("inf"),-1]:
            with self.assertRaises(ValueError):ComposeBudget(memory_mb=value)
        with self.assertRaises(ValueError):plan_projection({0:(20,20)},{0:np.full((3,3),np.nan)},20,"spherical",ComposeBudget())
        for roi in [(0,0,-1,2),(np.nan,0,2,2),(0,0,2**30,2)]:
            with self.assertRaises(ValueError):canvas_roi([roi])

    def test_small_budget_before_warp(self):
        with patch.object(cv2,'PyRotationWarper',wraps=cv2.PyRotationWarper) as warper:
            scale,_,rois,diag=plan_projection({0:(2000,1500)},{0:np.eye(3)},1500,'spherical',ComposeBudget(max_output_side=300,memory_mb=64))
            self.assertLess(scale,1);self.assertLessEqual(max(diag['canvas'][2:]),300)
            self.assertLessEqual(diag['estimated_peak_bytes'],64*1024**2)

    def test_plane_horizon(self):
        with self.assertRaises(ValueError):plan_projection({0:(720,540)},{0:np.diag([-1.,1.,-1.])},560,'plane',ComposeBudget())

    def test_mixed_work_resolution(self):
        sizes=[]
        class Backend:
            name='sift'
            def extract(self,img):
                sizes.append(img.shape[:2]);return ImageFeats(np.empty((0,2)),None,img.shape[1::-1])
            def match(self,*args):return (np.empty((0,2)),np.empty((0,2)))
        with patch('app.stitcher.pipeline.get_backend',return_value=Backend()):
            with self.assertRaises(StitchError):stitch_pro([np.zeros((1000,1000,3),np.uint8),np.zeros((4000,4000,3),np.uint8)],'sift')
        self.assertTrue(all(h*w<=601000 for h,w in sizes),sizes)

    def test_input_budget_rejects_before_resize(self):
        with patch.object(cv2,'resize') as resize:
            with self.assertRaises(StitchError):stitch_pro([np.zeros((100,100,3),np.uint8)]*2,'sift',memory_mb=.01)
            resize.assert_not_called()


class Reliability(unittest.TestCase):
    def test_coordinate_duplicates(self):
        f=ImageFeats(np.array([[1,1],[1,1],[2,2]],np.float32),None,(10,10))
        result=match_record(f,f,[(0,0),(1,1),(2,2)],[.9,.8,.7],'test')
        self.assertEqual(len(result.indices),2)

    def test_parallax_and_bad_local_match(self):
        xy=np.array([(x,y) for y in range(0,100,10) for x in range(0,100,10)],np.float32)
        residual=np.tile([12.,0.],(len(xy),1));residual[44]=[0,15]
        keep=localwarp.coherent_samples(xy,residual,80,30)
        self.assertFalse(keep[44]);self.assertGreaterEqual(keep.sum(),95)

    def test_field_fallback(self):
        mask=np.full((100,100),255,np.uint8);image=np.full((100,100,3),200,np.uint8)
        field=localwarp.OffsetField((0,0),(100,100),np.full((33,33,2),100,np.float32))
        with self.assertRaises(localwarp.UnsafeFieldError):localwarp.warp_with_field(image,mask,(0,0),field,1)

    def test_field_amplitude_gradient_and_sample_error(self):
        xy=np.array([(x,y) for y in range(-100,101,5) for x in range(-150,151,5)],np.float32)
        pair=_Pair(0,1,np.eye(3),xy,xy+np.float32([70,0]),1,all_i=xy,all_j=xy+np.float32([70,0]),scores=np.ones(len(xy)))
        warper=SimpleNamespace(warpPoint=lambda point,K,R:point)
        diag={}
        fields=localwarp.build_fields([0,1],[pair],{0:np.eye(3),1:np.eye(3)},warper,lambda i:np.eye(3),
                                     {0:(500,400),1:(500,400)},1,{0:(0,0,500,400),1:(0,0,500,400)},diagnostics=diag)
        for field in fields.values():
            self.assertLessEqual(np.linalg.norm(field.grid,axis=2).max(),24)
            gx=np.linalg.norm(np.diff(field.grid,axis=1),axis=2).max()/(500/32)
            gy=np.linalg.norm(np.diff(field.grid,axis=0),axis=2).max()/(400/32)
            self.assertLessEqual(gx+gy,.25001)
        residual=diag['local_warp']['expected_sample_residual']
        self.assertLess(residual['after_mean'],residual['before_mean'])

    def test_largest_component_precedes_weight(self):
        xy=np.zeros((4,2),np.float32)
        pair=lambda i,j,c:_Pair(i,j,np.eye(3),xy,xy,c)
        pairs=[pair(0,1,3)]+[pair(i,i+1,1) for i in range(2,6)]
        group=_components(7,pairs)[0];self.assertEqual(group,[2,3,4,5,6])
        selected=[p for p in pairs if p.i in group and p.j in group]
        root,_=_max_spanning_tree(7,selected);self.assertIn(root,group)

    def test_ba_acceptance(self):
        xy=np.array([(x,y) for y in [-10,0,10] for x in [-10,0,10]],np.float32)
        p=_Pair(0,1,np.eye(3),xy,xy,1)
        for mode in ['nan','worse','valid_limit']:
            def fake(fun,x0,**kwargs):
                x=x0.copy()
                if mode=='nan':x[0]=np.nan
                if mode=='worse':x[1]=.5
                return SimpleNamespace(x=x,status=0,success=False,nfev=200,message='limit')
            diag=[]
            with patch('scipy.optimize.least_squares',fake):_bundle_adjust([p],{0:np.eye(3),1:np.eye(3)},0,560,diag)
            self.assertEqual(diag[0]['accepted'],mode=='valid_limit')

    def test_empty_and_holey_crop(self):
        self.assertIsNone(autocrop.largest_inner_rect(np.zeros((8,8),np.uint8)))
        for shape in [(1280,1280),(1279,1003),(5,8)]:
            mask=np.full(shape,255,np.uint8);mask[min(301,shape[0]-2),min(301,shape[1]-2)]=0
            x,y,w,h=autocrop.largest_inner_rect(mask)
            self.assertTrue(np.all(mask[y:y+h,x:x+w]))

    def test_valid_edge_extension(self):
        mask=np.zeros((30,30),np.uint8);mask[5:25,5:25]=255
        image=np.zeros((30,30,3),np.uint8);image[mask>0]=240
        extended=extend_valid_colors(image,mask)
        self.assertTrue(np.all(extended==240));self.assertEqual(np.count_nonzero(mask),400)

    def test_core_has_no_web_import(self):
        for path in Path('app/stitcher').glob('*.py'):
            tree=ast.parse(path.read_text(encoding='utf-8'))
            for node in ast.walk(tree):
                if isinstance(node,ast.ImportFrom):self.assertFalse((node.module or '').startswith(('fastapi','starlette','app.main')))
                if isinstance(node,ast.Import):self.assertFalse(any(n.name.startswith(('fastapi','starlette','app.main')) for n in node.names))


if __name__=='__main__':unittest.main()
