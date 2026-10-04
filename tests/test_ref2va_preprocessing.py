"""Check video patch ordering against the official PIL image processor and audio gain."""
import sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from minimax_h3_mlx.ref2va import preprocess_reference_video, fft_resample
from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil

rng = np.random.default_rng(7)
image = rng.integers(0, 256, (64, 96, 3), dtype=np.uint8)
processor = Qwen2VLImageProcessorPil(patch_size=16, temporal_patch_size=2, merge_size=2, size={'shortest_edge':4096,'longest_edge':25165824}, image_mean=[0.5]*3, image_std=[0.5]*3)
expected = processor(images=[Image.fromarray(image)], return_tensors='np')
actual, grid = preprocess_reference_video(np.stack([image,image]))
np.testing.assert_array_equal(grid, expected['image_grid_thw'])
np.testing.assert_allclose(actual, expected['pixel_values'], atol=2e-7, rtol=0)
for source, target in [(48000,44100),(44100,48000),(8,16),(16,8)]:
    constant = np.full((2,source),0.375,np.float32)
    output = fft_resample(constant,source,target)
    assert output.shape == (2,target)
    np.testing.assert_allclose(output,0.375,atol=1e-6)
    t=np.arange(source)/source
    wave=np.stack([np.sin(2*np.pi*2*t)]*2).astype(np.float32)
    output=fft_resample(wave,source,target)
    expected_wave=np.stack([np.sin(2*np.pi*2*np.arange(target)/target)]*2)
    np.testing.assert_allclose(output,expected_wave,atol=1e-6)
print('Ref2VA patch order matches official PIL processor; FFT resampling preserves DC and sine amplitude')
# Load only the official standalone resize function; its containing processor imports torch.
import ast, math, transformers
source=Path(transformers.__file__).parent/'models/qwen3_vl/video_processing_qwen3_vl.py'
node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='smart_resize')
namespace={'math':math}
exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),namespace)
for t,h,w in [(20,768,1344),(8,2048,2048),(2,16,24)]:
    frames=np.zeros((t,h,w,3),np.uint8)
    _,grid=preprocess_reference_video(frames)
    expected_size=namespace['smart_resize'](t,h,w,factor=32,min_pixels=4096,max_pixels=25165824)
    assert tuple(grid[0,1:]*16)==expected_size,(grid,expected_size)
print('Video resize grid matches official Qwen3-VL total-frame pixel budget')
