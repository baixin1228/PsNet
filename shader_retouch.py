#!/usr/bin/env python3
"""独立 OpenGL 4.3 修图器。pip install -r requirements-shader-retouch.txt"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import moderngl

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.webp', '.bmp'}
PRESETS = {
    'vivid': dict(shadow=.45, highlight=.30, contrast=1.15, clarity=.25, saturation=.32, split=.0),
    'hdr': dict(shadow=.70, highlight=.50, contrast=1.20, clarity=.45, saturation=.38, split=.025),
    'cinematic': dict(shadow=.30, highlight=.40, contrast=1.12, clarity=.16, saturation=.18, split=.10),
}
HEADER = '''#version 430
layout(local_size_x=16,local_size_y=16) in;
layout(rgba16f,binding=0) readonly uniform image2D src;
layout(rgba16f,binding=1) writeonly uniform image2D dst;
ivec2 pos; ivec2 sz;
vec4 fetch(ivec2 p){return imageLoad(src,clamp(p,ivec2(0),sz-1));}
float lum(vec3 c){return dot(c,vec3(.2126,.7152,.0722));}
vec3 linearize(vec3 c){return mix(c/12.92,pow((c+.055)/1.055,vec3(2.4)),step(vec3(.04045),c));}
vec3 encode(vec3 c){c=max(c,vec3(0));return mix(12.92*c,1.055*pow(c,vec3(1./2.4))-.055,step(vec3(.0031308),c));}
'''

def shader(body):
    return HEADER + body

INIT = shader('''uniform float exposure;
void main(){pos=ivec2(gl_GlobalInvocationID.xy);sz=imageSize(src);if(any(greaterThanEqual(pos,sz)))return;
vec3 c=linearize(fetch(pos).rgb)*exp2(exposure);float l=log2(max(lum(c),1e-5));imageStore(dst,pos,vec4(l,l*l,0,1));}''').replace('layout(rgba16f,binding=0)', 'layout(rgba8,binding=0)')
BLUR = shader('''uniform ivec2 axis;uniform float radius;
void main(){pos=ivec2(gl_GlobalInvocationID.xy);sz=imageSize(src);if(any(greaterThanEqual(pos,sz)))return;
vec4 s=vec4(0);float w=0;for(int i=-12;i<=12;i++){float t=float(i)/12.;float k=exp(-4.5*t*t);s+=fetch(pos+axis*int(round(t*radius)))*k;w+=k;}imageStore(dst,pos,s/w);}''')
COEFF = shader('''void main(){pos=ivec2(gl_GlobalInvocationID.xy);sz=imageSize(src);if(any(greaterThanEqual(pos,sz)))return;
vec2 m=fetch(pos).rg;float v=max(m.y-m.x*m.x,0);float a=v/(v+.12);float b=(1-a)*m.x;imageStore(dst,pos,vec4(a,b,0,1));}''')
TONE = shader('''layout(rgba8,binding=2) readonly uniform image2D original;
uniform float shoulder;uniform float exposure;uniform float shadow;uniform float highlight;uniform float contrast;uniform float clarity;
void main(){pos=ivec2(gl_GlobalInvocationID.xy);sz=imageSize(original);if(any(greaterThanEqual(pos,sz)))return;
vec3 c=linearize(imageLoad(original,pos).rgb)*exp2(exposure);float y=max(lum(c),1e-5);float l=log2(y);
ivec2 gs=imageSize(src);vec2 uv=(vec2(pos)+.5)*vec2(gs)/vec2(sz)-.5;
ivec2 gp=ivec2(floor(uv));vec2 f=fract(uv);
vec2 ab=mix(mix(imageLoad(src,clamp(gp,ivec2(0),gs-1)).rg,imageLoad(src,clamp(gp+ivec2(1,0),ivec2(0),gs-1)).rg,f.x),mix(imageLoad(src,clamp(gp+ivec2(0,1),ivec2(0),gs-1)).rg,imageLoad(src,clamp(gp+ivec2(1,1),ivec2(0),gs-1)).rg,f.x),f.y);float base=ab.x*l+ab.y;float detail=l-base;
float lift=shadow*2.0*(1.-smoothstep(-5.,-1.,base));
float compress=highlight*max(base+1.5,0.);
float outY=exp2(base+lift-compress+detail+clarity*clamp(detail,-.6,.6));
c*=outY/y;
// A bounded shoulder preserves headroom; contrast acts on display luminance.
c=c/(1.+shoulder*c);vec3 s=encode(c);float sy=max(lum(s),1e-5);
float ny=1./(1.+exp(-contrast*log(clamp(sy,.0001,.9999)/(1.-clamp(sy,.0001,.9999)))));
s*=ny/sy;imageStore(dst,pos,vec4(s,1));}''')
COLOR = shader('''uniform float saturation;uniform float split;
vec3 toLab(vec3 c){vec3 v=mat3(.4122214708,.2119034982,.0883024619,.5363325363,.6806995451,.2817188376,.0514459929,.1073969566,.6299787005)*c;
v=pow(max(v,vec3(0)),vec3(1./3.));return mat3(.2104542553,1.9779984951,.0259040371,.7936177850,-2.4285922050,.7827717662,-.0040720468,.4505937099,-.8086757660)*v;}
vec3 fromLab(vec3 c){vec3 v=mat3(1.,1.,1.,.3963377774,-.1055613458,-.0894841775,.2158037573,-.0638541728,-1.2914855480)*c;v=v*v*v;return mat3(4.0767416621,-1.2684380046,-.0041960863,-3.3077115913,2.6097574011,-.7034186147,.2309699292,-.3413193965,1.7076147010)*v;}
void main(){pos=ivec2(gl_GlobalInvocationID.xy);sz=imageSize(src);if(any(greaterThanEqual(pos,sz)))return;
vec3 lab=toLab(linearize(max(fetch(pos).rgb,vec3(0))));float chroma=length(lab.yz);
float protection=smoothstep(.02,.18,lab.x)*(1.-smoothstep(.88,1.,lab.x));
lab.yz*=1.+saturation*(1.-.65*smoothstep(.08,.28,chroma))*protection;
float hue=atan(lab.z,lab.y);float blue=exp(-pow((hue+1.5)/.65,2.));lab.x-=saturation*.035*blue*protection;
lab.y+=split*.10;lab.z+=split*.35*(2.*smoothstep(.25,.85,lab.x)-1.);
// Compress chroma toward a neutral of the same lightness, instead of clipping channels.
vec3 c=fromLab(lab);if(any(lessThan(c,vec3(0)))||any(greaterThan(c,vec3(1)))){float lo=0.,hi=1.;for(int i=0;i<10;i++){float m=(lo+hi)*.5;vec3 t=fromLab(vec3(lab.x,lab.yz*m));if(all(greaterThanEqual(t,vec3(0)))&&all(lessThanEqual(t,vec3(1))))lo=m;else hi=m;}c=fromLab(vec3(lab.x,lab.yz*lo));}
imageStore(dst,pos,vec4(clamp(encode(c),0.,1.),1));}''')
SHARP = shader('''uniform float amount;
void main(){pos=ivec2(gl_GlobalInvocationID.xy);sz=imageSize(src);if(any(greaterThanEqual(pos,sz)))return;
vec3 c=fetch(pos).rgb;float y=lum(c);float blurred=0.;float weight=0.;
for(int j=-1;j<=1;j++)for(int i=-1;i<=1;i++){float k=float((i==0?2:1)*(j==0?2:1));blurred+=lum(fetch(pos+ivec2(i,j)).rgb)*k;weight+=k;}
float d=y-blurred/weight;
float edge=smoothstep(.002,.015,abs(d));float gain=amount*edge*smoothstep(.02,.15,y);
float ny=y+clamp(d,-.035,.035)*gain;c*=max(ny,0.)/max(y,1e-5);imageStore(dst,pos,vec4(clamp(c,0.,1.),1));}''').replace('layout(rgba16f,binding=1)', 'layout(rgba8,binding=1)')

class Retoucher:
    def __init__(self, backend='auto'):
        errors=[]
        for b in (['egl', None] if backend=='auto' else [None if backend=='default' else backend]):
            try:
                self.ctx=moderngl.create_standalone_context(require=430, **({'backend':b} if b else {}))
                break
            except Exception as e:
                errors.append(str(e))
        else:
            raise RuntimeError('无法创建 OpenGL 4.3 上下文：'+'; '.join(errors))
        self.programs={k:self.ctx.compute_shader(v) for k,v in dict(init=INIT,blur=BLUR,coeff=COEFF,tone=TONE,color=COLOR,sharp=SHARP).items()}

    def dispatch(self, name, source, dest, extra=None, **uniforms):
        source.bind_to_image(0,read=True,write=False)
        dest.bind_to_image(1,read=False,write=True)
        for binding, tex in (extra or {}).items():
            tex.bind_to_image(binding,read=True,write=False)
        p=self.programs[name]
        for k,v in uniforms.items():
            if k in p: p[k].value=v
        p.run((dest.width+15)//16,(dest.height+15)//16)
        self.ctx.memory_barrier()

    def process(self, image, style, strength, exposure_offset, sharpness):
        # Only small-image statistics run on CPU; all full-resolution enhancement runs in shaders.
        thumb=image.copy();thumb.thumbnail((256,256))
        rgb=np.asarray(thumb,dtype=np.float32)/255
        lin=np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)
        y=lin@np.array([.2126,.7152,.0722],dtype=np.float32)
        p10,p50,p95=np.quantile(y,[.1,.5,.95])
        ev=float(np.clip(math.log2(.18/max(float(p50),.015))*.55,-1.,1.5))
        # Limit brightening when the frame already contains extensive highlights.
        ev=min(ev,max(0.,math.log2(.95/max(float(p95),1e-5)))+.35) if ev>0 else ev
        ev=ev*strength+exposure_offset
        params={k:v*strength for k,v in PRESETS[style].items()}
        params['contrast']=1+(PRESETS[style]['contrast']-1)*strength
        w,h=image.size
        textures=[]
        timings={}
        def stamp(name, started):
            self.ctx.finish()
            timings[name]=time.perf_counter()-started
        def tex(size, data=None, dtype='f2'):
            t=self.ctx.texture(size,4,data,dtype=dtype);textures.append(t);return t
        def upload(im):
            arr=np.full((im.height,im.width,4),255,np.uint8)
            arr[:,:,:3]=np.asarray(im,dtype=np.uint8)
            return tex(im.size,arr.tobytes(),dtype='f1')
        try:
            t=time.perf_counter()
            original=upload(image)
            stamp('upload',t)
            t=time.perf_counter()
            analysis=image.copy();analysis.thumbnail((768,768),Image.Resampling.LANCZOS)
            small=upload(analysis)
            a,b,c,d=[tex(analysis.size) for _ in range(4)]
            self.dispatch('init',small,a,exposure=ev)
            radius=max(2.,min(analysis.size)*.025)
            self.dispatch('blur',a,b,axis=(1,0),radius=radius)
            self.dispatch('blur',b,c,axis=(0,1),radius=radius)
            self.dispatch('coeff',c,d)
            self.dispatch('blur',d,b,axis=(1,0),radius=radius)
            self.dispatch('blur',b,c,axis=(0,1),radius=radius)
            stamp('guide',t)
            t=time.perf_counter()
            full=tex((w,h))
            self.dispatch('tone',c,full,extra={2:original},shoulder=.22*strength,exposure=ev,**{k:params[k] for k in ['shadow','highlight','contrast','clarity']})
            stamp('tone',t)
            original.release();textures.remove(original)
            for buffer in [small,a,b,d]:buffer.release();textures.remove(buffer)
            c.release();textures.remove(c)
            t=time.perf_counter()
            colored=tex((w,h))
            self.dispatch('color',full,colored,saturation=params['saturation'],split=params['split'])
            stamp('color',t)
            full.release();textures.remove(full)
            t=time.perf_counter()
            output=tex((w,h),dtype='f1')
            self.dispatch('sharp',colored,output,amount=sharpness*strength)
            stamp('sharp',t)
            t=time.perf_counter()
            result=np.frombuffer(output.read(),np.uint8).reshape(h,w,4)[:,:,:3].copy()
            out=Image.fromarray(result)
            timings['readback']=time.perf_counter()-t
            if result.max()==0 and np.asarray(thumb).max()>16:
                raise RuntimeError('GPU 输出全黑，已停止保存。renderer='+self.ctx.info['GL_RENDERER'])
            error=self.ctx.error
            if error!='GL_NO_ERROR':raise RuntimeError('OpenGL error: '+error)
            return out,dict(exposure_ev=ev,brightness_quantiles=[float(p10),float(p50),float(p95)],timings=timings,**params)
        finally:
            for t in textures:t.release()


def main():
    ap=argparse.ArgumentParser(description='GPU shader 自动修图；单图输出到原目录，文件夹输出到同级 <目录名>_retouched。输入应为 sRGB。')
    ap.add_argument('input',type=Path)
    ap.add_argument('--style',choices=PRESETS,default='vivid')
    ap.add_argument('--strength',type=float,default=1.,help='整体强度 0–2；0 为原图（曝光补偿也不执行）')
    ap.add_argument('--exposure',type=float,default=0.,help='额外曝光补偿 EV，-3 到 3')
    ap.add_argument('--sharpness',type=float,default=.6,help='锐化强度 0–2')
    ap.add_argument('--max-size',type=int,default=0,help='最长边；0 保留原尺寸，256 适合当前小图训练')
    ap.add_argument('--recursive',action='store_true',help='递归目录，保留子目录结构')
    ap.add_argument('--overwrite',action='store_true')
    ap.add_argument('--backend',choices=['auto','egl','default'],default='auto')
    ap.add_argument('--allow-software',action='store_true',help='允许 Mesa 软件渲染，仅用于测试')
    args=ap.parse_args()
    if not all(math.isfinite(x) for x in [args.strength,args.exposure,args.sharpness]) or not 0<=args.strength<=2 or not 0<=args.sharpness<=2 or not -3<=args.exposure<=3 or args.max_size<0:
        ap.error('检查参数范围：strength/sharpness 0–2，exposure -3–3，max-size >= 0')
    root=args.input.expanduser().resolve()
    if root.is_file() and root.suffix.lower() in EXTENSIONS:
        files=[root];output=root.parent
    elif root.is_dir():
        files=sorted(p for p in (root.rglob('*') if args.recursive else root.iterdir()) if p.is_file() and p.suffix.lower() in EXTENSIONS and not p.stem.endswith('_retouched'))
        output=root.with_name(root.name+'_retouched')
    else:ap.error('输入不是支持的图片或文件夹')
    if not files:ap.error('未找到支持的图片')
    try: engine=Retoucher(args.backend)
    except Exception as e:ap.exit(1,str(e)+'\n')
    renderer=engine.ctx.info['GL_RENDERER'];print('OpenGL renderer:',renderer)
    if any(s in renderer.lower() for s in ['llvmpipe','softpipe','software','swrast']) and not args.allow_software:
        ap.exit(1,'检测到 CPU 软件渲染；请配置 GPU 驱动。仅测试可加 --allow-software。\n')
    failed=0
    for path in files:
        relative=path.relative_to(root) if root.is_dir() else Path(path.name)
        # Keep original extensions in folder mode to prevent same-stem source collisions.
        dest=(output/relative).with_name(relative.name+'_retouched.jpg') if root.is_dir() else path.with_name(path.stem+'_retouched.jpg')
        metadata=dest.with_suffix('.json')
        if (dest.exists() or metadata.exists()) and not args.overwrite:
            print('跳过已存在:',dest);continue
        start=time.perf_counter()
        try:
            with Image.open(path) as im:
                if getattr(im,'n_frames',1)>1:raise ValueError('不支持多帧图片')
                if 'A' in im.getbands() or 'transparency' in im.info:raise ValueError('输入包含透明度，请先明确背景并转换 RGB')
                if im.info.get('icc_profile'):
                    import io
                    from PIL import ImageCms
                    im=ImageCms.profileToProfile(im,ImageCms.ImageCmsProfile(io.BytesIO(im.info['icc_profile'])),ImageCms.createProfile('sRGB'),outputMode='RGB')
                image=ImageOps.exif_transpose(im).convert('RGB')
            if args.max_size:image.thumbnail((args.max_size,args.max_size),Image.Resampling.LANCZOS)
            if args.strength==0:result,params=image.copy(),{}
            else:result,params=engine.process(image,args.style,args.strength,args.exposure,args.sharpness)
            dest.parent.mkdir(parents=True,exist_ok=True)
            from PIL import ImageCms
            srgb=ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
            save_started=time.perf_counter()
            result.save(dest,format="JPEG",quality=85,subsampling=2,icc_profile=srgb)
            save_seconds=time.perf_counter()-save_started
            record=dict(algorithm='shader-retouch-v2',input=str(path),output=str(dest),size=list(image.size),style=args.style,strength=args.strength,sharpness=args.sharpness,exposure_offset=args.exposure,renderer=renderer,parameters=params,save_seconds=save_seconds,seconds=time.perf_counter()-start)
            metadata.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
            print(f'完成 {record["seconds"]:.2f}s: {dest}')
            if 'timings' in params:print('  分段耗时:', ', '.join(f'{k}={v:.3f}s' for k,v in params['timings'].items()), f', save={save_seconds:.3f}s')
        except Exception as e:
            failed+=1;print(f'失败 {path}: {e}',file=sys.stderr)
    engine.ctx.release()
    return 1 if failed else 0

if __name__=='__main__':sys.exit(main())
