"""The conditioning loader must preserve encoder values while omitting decoder residency."""
import sys,json,tempfile
from pathlib import Path
import mlx.core as mx
from mlx.utils import tree_flatten
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from minimax_h3_mlx.load import read_video_vae_config,load_video_vae
from minimax_h3_mlx.video_vae import VideoVAE
with tempfile.TemporaryDirectory() as folder:
    root=Path(folder);(root/'source').mkdir()
    source={'ch':32,'ch_mult':[1],'in_channels':3,'out_ch':3,'z_channels':4,'num_res_blocks':1,'space_down':[2],'time_down':[1],'vit_decoder_kwargs':{'num_layers':2,'heads':2,'dim_head':12,'rope_theta':10000,'rope_dim_ratio':0.5}}
    (root/'source/config.json').write_text(json.dumps(source))
    (root/'config.json').write_text(json.dumps({'vae_clip_length':5,'vae_token_drop':1}))
    mx.random.seed(0)
    model=VideoVAE(read_video_vae_config(root))
    weights={k:(v.transpose(0,4,1,2,3) if v.ndim==5 else v) for k,v in tree_flatten(model.parameters())}
    mx.save_safetensors(str(root/'source/model.safetensors'),weights)
    full=load_video_vae(root);slim=load_video_vae(root,encode_only=True)
    keys=dict(tree_flatten(slim.parameters()))
    assert not any(k.startswith(('decoder.','post_quant_conv.')) for k in keys)
    full_weights=dict(tree_flatten(full.parameters()))
    for k,v in keys.items():
        assert bool(mx.array_equal(v,full_weights[k]).item()),k
    x=mx.random.normal((1,3,7,16,16))
    a=full.encode(x);b=slim.encode(x);mx.eval(a,b)
    assert bool(mx.array_equal(a,b).item())
    try:slim.decode(mx.zeros((1,4,9,8,8)))
    except RuntimeError as exc:assert 'encoding only' in str(exc)
    else:raise AssertionError('encode-only decoder must fail explicitly')
print('Encoder-only checkpoint load has identical encoder weights and output; decoder parameters absent')
