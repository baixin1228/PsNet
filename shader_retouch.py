#!/usr/bin/env python3
"""独立 OpenGL 4.3 修图器。pip install -r requirements-shader-retouch.txt"""
import argparse
import json
import math
import sys
import time
import queue
import os
import threading
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import shared_memory
from collections import deque, OrderedDict
from dataclasses import dataclass
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
vec3 c=linearize(fetch(pos).rgb)*exp2(exposure);float l=log2(max(lum(c),1e-5));imageStore(dst,pos,vec4(l,l*l,0,1));}''').replace('layout(rgba16f,binding=0) readonly uniform image2D src;', 'layout(binding=0) uniform sampler2D src;').replace('imageSize(src)', 'textureSize(src,0)').replace('imageLoad(src,clamp(p,ivec2(0),sz-1))','texelFetch(src,clamp(p,ivec2(0),sz-1),0)')
# Exposure remains in an SSBO throughout the GPU pipeline (no CPU round trip).
EXPOSURE_DECL = 'layout(std430,binding=4) readonly buffer Exposure { vec4 exposureStats; };\n'
INIT=INIT.replace('uniform float exposure;',EXPOSURE_DECL).replace('exp2(exposure)','exp2(exposureStats.x)')
DOWNSAMPLE = INIT.split('void main(){',1)[0]+"""void main(){
 pos=ivec2(gl_GlobalInvocationID.xy);sz=imageSize(dst);if(any(greaterThanEqual(pos,sz)))return;
 vec2 uv=(vec2(pos)+.5)/vec2(sz);vec2 ratio=vec2(textureSize(src,0))/vec2(sz);
 float lod=max(0.,log2(max(ratio.x,ratio.y)));
 imageStore(dst,pos,textureLod(src,uv,lod));} """
HISTOGRAM = shader("""layout(std430,binding=3) buffer Histogram {uint bins[4096];uint maxValue;};
void main(){pos=ivec2(gl_GlobalInvocationID.xy);sz=imageSize(src);if(any(greaterThanEqual(pos,sz)))return;
 vec3 c=fetch(pos).rgb;float y=max(lum(linearize(c)),exp2(-16.));
 int index=clamp(int((log2(y)+16.)*256.),0,4095);atomicAdd(bins[index],1u);
 atomicMax(maxValue,uint(round(max(c.r,max(c.g,c.b))*255.)));}
""")
EXPOSURE = """#version 430
layout(local_size_x=1) in;
layout(std430,binding=3) readonly buffer Histogram {uint bins[4096];uint maxValue;};
layout(std430,binding=4) writeonly buffer Exposure {vec4 exposureStats;};
uniform uint count;uniform float strength;uniform float offset;
float quantile(float q){float rank=q*float(count-1u);uint acc=0u;
 for(int i=0;i<4096;i++){uint n=bins[i];if(float(acc+n)>rank)return exp2(-16.+(float(i)+.5)/256.);acc+=n;}return 1.;}
void main(){float p10=quantile(.1),p50=quantile(.5),p95=quantile(.95);
 float ev=clamp(log2(.18/max(p50,.015))*.55,-1.,1.5);
 if(ev>0.)ev=min(ev,max(0.,log2(.95/max(p95,1e-5)))+.35);
 exposureStats=vec4(ev*strength+offset,p10,p50,p95);}
"""
BLUR = shader('''uniform ivec2 axis;uniform float radius;
void main(){pos=ivec2(gl_GlobalInvocationID.xy);sz=imageSize(src);if(any(greaterThanEqual(pos,sz)))return;
vec4 s=vec4(0);float w=0;for(int i=-12;i<=12;i++){float t=float(i)/12.;float k=exp(-4.5*t*t);s+=fetch(pos+axis*int(round(t*radius)))*k;w+=k;}imageStore(dst,pos,s/w);}''')
COEFF = shader('''void main(){pos=ivec2(gl_GlobalInvocationID.xy);sz=imageSize(src);if(any(greaterThanEqual(pos,sz)))return;
vec2 m=fetch(pos).rg;float v=max(m.y-m.x*m.x,0);float a=v/(v+.12);float b=(1-a)*m.x;imageStore(dst,pos,vec4(a,b,0,1));}''')
TONE = shader('''layout(binding=2) uniform sampler2D original;
uniform float look_rolloff;uniform float exposure;uniform float shadow;uniform float highlight;uniform float contrast;uniform float clarity;
void main(){pos=ivec2(gl_GlobalInvocationID.xy);sz=textureSize(original,0);if(any(greaterThanEqual(pos,sz)))return;
vec3 c=linearize(texelFetch(original,pos,0).rgb)*exp2(exposure);float y=max(lum(c),1e-5);float l=log2(y);
ivec2 gs=imageSize(src);vec2 uv=(vec2(pos)+.5)*vec2(gs)/vec2(sz)-.5;
ivec2 gp=ivec2(floor(uv));vec2 f=fract(uv);
vec2 ab=mix(mix(imageLoad(src,clamp(gp,ivec2(0),gs-1)).rg,imageLoad(src,clamp(gp+ivec2(1,0),ivec2(0),gs-1)).rg,f.x),mix(imageLoad(src,clamp(gp+ivec2(0,1),ivec2(0),gs-1)).rg,imageLoad(src,clamp(gp+ivec2(1,1),ivec2(0),gs-1)).rg,f.x),f.y);float base=ab.x*l+ab.y;float detail=l-base;
float lift=shadow*2.0*(1.-smoothstep(-5.,-1.,base));
float compress=highlight*max(base+1.5,0.);
float outY=exp2(base+lift-compress+detail+clarity*clamp(detail,-.6,.6));
c*=outY/y;
// Restore the original tonal look without clipping extended-range values.
c=c/(1.+look_rolloff*c);
// Keep linear extended-range values until the final color/output mapping.
// Extend the existing contrast curve above .8 without clamping highlights.
vec3 display=encode(c);float sy=max(lum(display),1e-5);
float pivot=.8;
float atPivot=1./(1.+exp(-contrast*log(pivot/(1.-pivot))));
float slope=contrast*atPivot*(1.-atPivot)/(pivot*(1.-pivot));
float ny;
if(sy<pivot){float v=clamp(sy,.0001,.9999);ny=1./(1.+exp(-contrast*log(v/(1.-v))));}
else ny=atPivot+slope*(sy-pivot);
display*=ny/sy;
imageStore(dst,pos,vec4(linearize(display),1));}''')
TONE=TONE.replace('uniform float exposure;',EXPOSURE_DECL).replace('exp2(exposure)','exp2(exposureStats.x)')
COLOR = shader('''uniform float saturation;uniform float split;uniform float stylize;
float angleDelta(float a,float b){return atan(sin(a-b),cos(a-b));}
float hueWeight(float h,float center,float width){float d=angleDelta(h,center)/width;return exp(-.5*d*d);}
vec3 toLab(vec3 c){vec3 v=mat3(.4122214708,.2119034982,.0883024619,.5363325363,.6806995451,.2817188376,.0514459929,.1073969566,.6299787005)*c;
v=pow(max(v,vec3(0)),vec3(1./3.));return mat3(.2104542553,1.9779984951,.0259040371,.7936177850,-2.4285922050,.7827717662,-.0040720468,.4505937099,-.8086757660)*v;}
vec3 fromLab(vec3 c){vec3 v=mat3(1.,1.,1.,.3963377774,-.1055613458,-.0894841775,.2158037573,-.0638541728,-1.2914855480)*c;v=v*v*v;return mat3(4.0767416621,-1.2684380046,-.0041960863,-3.3077115913,2.6097574011,-.7034186147,.2309699292,-.3413193965,1.7076147010)*v;}
void main(){pos=ivec2(gl_GlobalInvocationID.xy);sz=imageSize(src);if(any(greaterThanEqual(pos,sz)))return;
vec3 lab=toLab(max(fetch(pos).rgb,vec3(0)));float chroma=length(lab.yz);
float protection=smoothstep(.02,.18,lab.x)*(1.-smoothstep(.88,1.,lab.x));
lab.yz*=1.+saturation*(1.-.65*smoothstep(.08,.28,chroma))*protection;
float hue=atan(lab.z,lab.y);float blue=exp(-pow((hue+1.5)/.65,2.));lab.x-=saturation*.035*blue*protection;
lab.y+=split*.10;lab.z+=split*.35*(2.*smoothstep(.25,.85,lab.x)-1.);
// Optional smooth palette shaping; never quantize hues or tint neutral pixels.
if(stylize>0.){
 float C=length(lab.yz);float h=atan(lab.z,lab.y);
 float mask=smoothstep(.018,.065,C)*smoothstep(.08,.25,lab.x)*(1.-smoothstep(.88,.99,lab.x));
 // Approximate skin-color protection; orange objects share this protection.
 float skin=hueWeight(h,.88,.36)*smoothstep(.25,.45,lab.x)*(1.-smoothstep(.85,.98,lab.x));
 float k=stylize*mask*(1.-.75*skin);
 // OKLCh hue anchors in radians: red, orange, yellow, green, cyan, blue, purple, magenta.
 const float centers[8]=float[8](.44,1.05,1.66,2.36,-2.88,-1.83,-.96,-.26);
 const float targets[8]=float[8](.51,.94,1.57,2.60,-2.70,-1.99,-.79,-.12);
 const float boosts[8]=float[8](.28,.22,.20,.34,.30,.38,.34,.30);
 const float lights[8]=float[8](-.008,.012,.015,-.012,-.010,-.035,.008,.005);
 float shift=0.;float boost=0.;float light=0.;float weights=0.;
 for(int i=0;i<8;i++){
  float w=hueWeight(h,centers[i],.31);
  shift+=w*angleDelta(targets[i],h);boost+=w*boosts[i];light+=w*lights[i];weights+=w;
 }
 float norm=max(weights,1.);
 // Smooth bounded rotation retains gradients instead of snapping to palette entries.
 h+=clamp(.80*k*shift/norm,-.40,.40);
 C*=1.+k*boost/norm*(1.-.65*smoothstep(.16,.30,C));
 lab.x+=k*light/norm;
 lab.yz=C*vec2(cos(h),sin(h));
}
// Preserve linear extended-range color for final SDR mapping.
imageStore(dst,pos,vec4(fromLab(lab),1));}''')
MAP = COLOR.split('void main(){',1)[0]+'''
void main(){pos=ivec2(gl_GlobalInvocationID.xy);sz=imageSize(src);if(any(greaterThanEqual(pos,sz)))return;
vec3 c=fetch(pos).rgb;float y=max(lum(c),0.);
// Identity in shadows/midtones; smoothly roll only the upper luminance range.
float knee=.72;float ceiling=.995;
if(y>knee){float mapped=knee+(ceiling-knee)*(1.-exp(-(y-knee)/(ceiling-knee)));c*=mapped/max(y,1e-5);}
vec3 lab=toLab(c);
// Bright highlights gently converge toward neutral, leaving midtone color intact.
float highlight=smoothstep(.80,.99,max(lum(c),0.));
lab.yz*=1.-.18*highlight;c=fromLab(lab);
// Fit only this pixel's chroma to the SDR gamut; never darken the whole frame.
if(any(lessThan(c,vec3(0)))||any(greaterThan(c,vec3(1)))){
 float lo=0.,hi=1.;
 for(int i=0;i<12;i++){float m=(lo+hi)*.5;vec3 t=fromLab(vec3(lab.x,lab.yz*m));
 if(all(greaterThanEqual(t,vec3(0)))&&all(lessThanEqual(t,vec3(1))))lo=m;else hi=m;}
 c=fromLab(vec3(lab.x,lab.yz*lo));
}
imageStore(dst,pos,vec4(clamp(encode(c),0.,1.),1));}'''
SHARP = shader('''uniform float amount;
void main(){pos=ivec2(gl_GlobalInvocationID.xy);sz=imageSize(src);if(any(greaterThanEqual(pos,sz)))return;
vec3 c=fetch(pos).rgb;float y=lum(c);float blurred=0.;float weight=0.;
for(int j=-1;j<=1;j++)for(int i=-1;i<=1;i++){float k=float((i==0?2:1)*(j==0?2:1));blurred+=lum(fetch(pos+ivec2(i,j)).rgb)*k;weight+=k;}
float d=y-blurred/weight;
float edge=smoothstep(.002,.015,abs(d));float gain=amount*edge*smoothstep(.02,.15,y);
float adjustment=clamp(d,-.035,.035)*gain;
float peak=max(c.r,max(c.g,c.b));
if(adjustment>0.)adjustment=min(adjustment,max(0.,(1.-peak)*y/max(peak,1e-5))*.9);
float ny=y+adjustment;c*=max(ny,0.)/max(y,1e-5);imageStore(dst,pos,vec4(clamp(c,0.,1.),1));}''').replace('layout(rgba16f,binding=1)', 'layout(rgba8,binding=1)')

# Validate output on the GPU, avoiding a full CPU scan before JPEG encoding.
SHARP=SHARP.replace('void main(){pos=ivec2(gl_GlobalInvocationID.xy);sz=imageSize(src);if(any(greaterThanEqual(pos,sz)))return;',
    'vec3 sharpenAt(ivec2 pixel){pos=pixel;sz=imageSize(src);')
SHARP=SHARP.replace('imageStore(dst,pos,vec4(clamp(c,0.,1.),1));}', 'return clamp(c,0.,1.);}')
SHARP+="""
layout(std430,binding=6) buffer OutputCheck {uint nonzero;};
shared uint maxima[256];
void main(){ivec2 pixel=ivec2(gl_GlobalInvocationID.xy);ivec2 size=imageSize(src);
 bool valid=all(lessThan(pixel,size));vec3 c=vec3(0);
 if(valid){c=sharpenAt(pixel);imageStore(dst,pixel,vec4(c,1));}
 uint lane=gl_LocalInvocationIndex;maxima[lane]=uint(round(max(c.r,max(c.g,c.b))*255.));
 barrier();for(uint stride=128u;stride>0u;stride>>=1u){if(lane<stride)maxima[lane]=max(maxima[lane],maxima[lane+stride]);barrier();}
 if(lane==0u)atomicMax(nonzero,maxima[0]);}
"""

def prepare_image(image):
    """Readers decode/pack only; analysis and exposure are computed in shaders."""
    return dict(original_bytes=image.tobytes())


def read_image(path, max_size):
    from PIL import ImageCms
    import io
    with Image.open(path) as im:
        if getattr(im,'n_frames',1)>1:raise ValueError('不支持多帧图片')
        if 'A' in im.getbands() or 'transparency' in im.info:raise ValueError('输入包含透明度，请先明确背景并转换 RGB')
        ImageOps.exif_transpose(im,in_place=True)
        image=im
        if im.info.get('icc_profile'):
            image=ImageCms.profileToProfile(image,ImageCms.ImageCmsProfile(io.BytesIO(im.info['icc_profile'])),ImageCms.createProfile('sRGB'),outputMode='RGB')
        else:image=image.convert('RGB')
    if max_size:image.thumbnail((max_size,max_size),Image.Resampling.LANCZOS)
    return image


def decode_shared(path, max_size):
    """Runs in a separate CPU process; only a small descriptor crosses the pipe."""
    started=time.perf_counter()
    image=read_image(path,max_size)
    decoded=time.perf_counter()
    shm=shared_memory.SharedMemory(create=True,size=image.width*image.height*3)
    try:
        shm.buf[:]=image.tobytes()
        return (shm.name,image.size,{'decode_resize':decoded-started,'shared_pack':time.perf_counter()-decoded})
    except BaseException:
        shm.unlink();raise
    finally:shm.close()


class SharedInput:
    def __init__(self, descriptor):
        name,self.size,self.decode_timings=descriptor
        self.shm=shared_memory.SharedMemory(name=name)

    def copy(self):
        return Image.frombytes('RGB',self.size,bytes(self.shm.buf))

    def close(self):
        if self.shm is not None:
            self.shm.close()
            try:self.shm.unlink()
            except FileNotFoundError:pass
            self.shm=None



@dataclass
class ReadbackPixels:
    data: bytes
    size: tuple
    check_black: bool

    def image(self):
        result=Image.frombuffer('RGBX',self.size,self.data,'raw','RGBX',0,1)
        if self.check_black and max(channel[1] for channel in result.getextrema()[:3])==0:
            raise RuntimeError('GPU 输出全黑，已停止保存')
        return result


class PendingReadback:
    def __init__(self, buffer, size, check_black, exposure=None, params=None, histogram=None, output_check=None):
        self.buffer=buffer
        self.size=size
        self.check_black=check_black
        self.exposure=exposure
        self.histogram=histogram
        self.output_check=output_check
        self.params=params

    def resolve(self):
        try:
            data=self.buffer.read()
            if self.exposure is not None:
                values=np.frombuffer(self.exposure.read(),np.float32)
                self.params.update(exposure_ev=float(values[0]),brightness_quantiles=values[1:4].tolist())
            check_black=self.check_black
            if self.histogram is not None:check_black=int.from_bytes(self.histogram.read(size=4,offset=4096*4),'little')>16
            if check_black and self.output_check is not None and int.from_bytes(self.output_check.read(),'little')==0:
                raise RuntimeError('GPU 输出全黑，已停止保存')
            return ReadbackPixels(data,self.size,False if self.output_check is not None else check_black)
        finally:self.release()

    def release(self):
        if self.buffer is not None:
            self.buffer.release();self.buffer=None
        if self.exposure is not None:
            self.exposure.release();self.exposure=None
        if self.histogram is not None:
            self.histogram.release();self.histogram=None
        if self.output_check is not None:
            self.output_check.release();self.output_check=None


class Retoucher:
    def __init__(self, backend='auto', profile=False, texture_cache_mb=512):
        self.profile=profile
        self.texture_budget=texture_cache_mb*1024*1024
        self.texture_pool=OrderedDict()
        self.texture_pool_bytes=0
        self.texture_hits=0
        self.texture_misses=0
        self.texture_evictions=0
        errors=[]
        for b in (['egl', None] if backend=='auto' else [None if backend=='default' else backend]):
            try:
                self.ctx=moderngl.create_standalone_context(require=430, **({'backend':b} if b else {}))
                break
            except Exception as e:
                errors.append(str(e))
        else:
            raise RuntimeError('无法创建 OpenGL 4.3 上下文：'+'; '.join(errors))
        self.programs={k:self.ctx.compute_shader(v) for k,v in dict(init=INIT,downsample=DOWNSAMPLE,histogram=HISTOGRAM,exposure=EXPOSURE,blur=BLUR,coeff=COEFF,tone=TONE,color=COLOR,map=MAP,sharp=SHARP).items()}

    @staticmethod
    def texture_bytes(texture):
        # RGB input textures own a mip chain after analysis.
        base=texture.width*texture.height*texture.components*int(texture.dtype[1:])
        return (base*4+2)//3 if texture.components==3 else base

    def acquire_texture(self,size,components,dtype,data=None):
        key=(size,components,dtype)
        found=None
        for identifier,(candidate,texture) in reversed(self.texture_pool.items()):
            if candidate==key:found=identifier;break
        if found is None:
            self.texture_misses+=1
            return self.ctx.texture(size,components,data,dtype=dtype,alignment=1)
        _,texture=self.texture_pool.pop(found)
        self.texture_pool_bytes-=self.texture_bytes(texture)
        self.texture_hits+=1
        if data is not None:texture.write(data,alignment=1)
        return texture

    def recycle_texture(self,texture):
        cost=self.texture_bytes(texture)
        if cost>self.texture_budget:
            texture.release();return
        while self.texture_pool and self.texture_pool_bytes+cost>self.texture_budget:
            _,(_,old)=self.texture_pool.popitem(last=False)
            self.texture_pool_bytes-=self.texture_bytes(old)
            self.texture_evictions+=1;old.release()
        self.texture_pool[texture.glo]=((texture.size,texture.components,texture.dtype),texture)
        self.texture_pool_bytes+=cost

    def close(self):
        for _,texture in self.texture_pool.values():texture.release()
        self.texture_pool.clear();self.texture_pool_bytes=0
        self.ctx.release()

    def dispatch(self, name, source, dest, extra=None, **uniforms):
        if source.components==3 or name=='init':source.use(location=0)
        else:source.bind_to_image(0,read=True,write=False)
        dest.bind_to_image(1,read=False,write=True)
        for binding, tex in (extra or {}).items():
            if tex.components==3:tex.use(location=binding)
            else:tex.bind_to_image(binding,read=True,write=False)
        p=self.programs[name]
        for k,v in uniforms.items():
            if k in p: p[k].value=v
        p.run((dest.width+15)//16,(dest.height+15)//16)
        self.ctx.memory_barrier()

    def process(self, image, style, strength, exposure_offset, sharpness, stylize=False, prepared=None, defer_readback=False):
        for identifier,(_,texture) in list(self.texture_pool.items()):
            if max(texture.size)>768 and texture.size!=image.size:
                self.texture_pool.pop(identifier)
                self.texture_pool_bytes-=self.texture_bytes(texture)
                self.texture_evictions+=1;texture.release()
        prepared=prepare_image(image) if prepared is None else prepared
        params={k:v*strength for k,v in PRESETS[style].items()}
        params['contrast']=1+(PRESETS[style]['contrast']-1)*strength
        w,h=image.size
        textures=[]
        buffers=[]
        timings={}
        allocation_seconds=0.
        allocation_count=0
        def stamp(name, started):
            if self.profile:self.ctx.finish()
            timings[name]=time.perf_counter()-started
        def tex(size, data=None, dtype='f2', components=4):
            nonlocal allocation_seconds,allocation_count
            started=time.perf_counter()
            t=self.acquire_texture(size,components,dtype,data)
            allocation_seconds+=time.perf_counter()-started;allocation_count+=1
            textures.append(t);return t
        try:
            t=time.perf_counter()
            original=tex(image.size,prepared['original_bytes'],dtype='f1',components=3)
            stamp('upload',t)
            t=time.perf_counter()
            original.build_mipmaps()
            scale=min(1.,768./max(w,h));analysis_size=(max(1,round(w*scale)),max(1,round(h*scale)))
            small=tex(analysis_size)
            self.dispatch('downsample',original,small)
            stats_scale=min(1.,256./max(w,h));stats_size=(max(1,round(w*stats_scale)),max(1,round(h*stats_scale)))
            stats=tex(stats_size)
            self.dispatch('downsample',original,stats)
            histogram=self.ctx.buffer(reserve=4097*4);buffers.append(histogram);histogram.clear()
            exposure=self.ctx.buffer(reserve=16);buffers.append(exposure)
            histogram.bind_to_storage_buffer(3);exposure.bind_to_storage_buffer(4)
            stats.bind_to_image(0,read=True,write=False)
            self.programs['histogram'].run((stats_size[0]+15)//16,(stats_size[1]+15)//16)
            self.ctx.memory_barrier()
            ep=self.programs['exposure'];ep['count'].value=stats_size[0]*stats_size[1]
            ep['strength'].value=strength;ep['offset'].value=exposure_offset
            ep.run();self.ctx.memory_barrier()
            self.recycle_texture(stats);textures.remove(stats)
            a,b,c,d=[tex(analysis_size) for _ in range(4)]
            self.dispatch('init',small,a)
            radius=max(2.,min(analysis_size)*.025)
            self.dispatch('blur',a,b,axis=(1,0),radius=radius)
            self.dispatch('blur',b,c,axis=(0,1),radius=radius)
            self.dispatch('coeff',c,d)
            self.dispatch('blur',d,b,axis=(1,0),radius=radius)
            self.dispatch('blur',b,c,axis=(0,1),radius=radius)
            stamp('guide',t)
            t=time.perf_counter()
            full=tex((w,h))
            self.dispatch('tone',c,full,extra={2:original},look_rolloff=.22*strength,**{k:params[k] for k in ['shadow','highlight','contrast','clarity']})
            stamp('tone',t)
            self.recycle_texture(original);textures.remove(original)
            for buffer in [small,a,b,d]:self.recycle_texture(buffer);textures.remove(buffer)
            self.recycle_texture(c);textures.remove(c)
            t=time.perf_counter()
            colored=tex((w,h))
            self.dispatch('color',full,colored,saturation=params['saturation'],split=params['split'],stylize=min(strength,1.5) if stylize else 0.)
            stamp('color',t)
            t=time.perf_counter()
            self.dispatch('map',colored,full)
            stamp('output_sdr',t)
            self.recycle_texture(colored);textures.remove(colored)
            t=time.perf_counter()
            output=tex((w,h),dtype='f1')
            output_check=self.ctx.buffer(reserve=4);buffers.append(output_check);output_check.clear()
            output_check.bind_to_storage_buffer(6)
            self.dispatch('sharp',full,output,amount=sharpness*strength)
            stamp('sharp',t)
            t=time.perf_counter()
            if defer_readback:
                # GPU pixel-pack buffer: enqueue transfer, submit the next image before mapping it.
                pack=self.ctx.buffer(reserve=w*h*4)
                try:
                    output.read_into(pack)
                except BaseException:
                    pack.release();raise
                out=PendingReadback(pack,(w,h),False,exposure,params,histogram,output_check)
                buffers.remove(output_check)
                buffers.remove(histogram)
                buffers.remove(exposure)
                timings['readback_enqueue']=time.perf_counter()-t
            else:
                raw=output.read()
                check_black=int.from_bytes(histogram.read(size=4,offset=4096*4),'little')>16
                if check_black and int.from_bytes(output_check.read(),'little')==0:raise RuntimeError('GPU 输出全黑，已停止保存')
                out=ReadbackPixels(raw,(w,h),False).image()
                values=np.frombuffer(exposure.read(),np.float32)
                params.update(exposure_ev=float(values[0]),brightness_quantiles=values[1:4].tolist())
                timings['readback']=time.perf_counter()-t
            error=self.ctx.error
            if error!='GL_NO_ERROR':
                if isinstance(out,PendingReadback):out.release()
                raise RuntimeError('OpenGL error: '+error)
            params.update(texture_create_upload_seconds=allocation_seconds,texture_count=allocation_count,timings=timings,timing_mode='synchronized' if self.profile else 'cpu_submission_readback_includes_gpu_wait')
            return out,params
        finally:
            for t in textures:self.recycle_texture(t)
            for buffer in buffers:buffer.release()


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
    ap.add_argument('--decode-backend',choices=['process','thread'],default='process',help='批量默认独立进程解码并通过共享内存传图；thread 可作对照，单图自动用线程')
    ap.add_argument('--read-workers',type=int,default=min(4,os.cpu_count() or 1),help='读图/预处理线程数，默认最多 4')
    ap.add_argument('--write-workers',type=int,default=2,help='JPEG 编码/保存线程数，默认 2')
    ap.add_argument('--queue-size',type=int,default=2,help='每级等待队列容量，默认 2；大图注意内存')
    ap.add_argument('--gpu-queue',type=int,default=2,help='GPU 在途读回数量，默认 2；1 可用作同步对照')
    ap.add_argument('--texture-cache-mb',type=int,default=512,help='空闲纹理LRU缓存上限MiB；默认512，0禁用用于对照')
    ap.add_argument('--diagnostics',action='store_true',help='打印队列、CPU阶段和GPU提交/读回诊断；不强制GPU同步')
    ap.add_argument('--profile-gpu',action='store_true',help='同步分段计时；会降低流水线吞吐量')
    ap.add_argument('--stylize',action='store_true',help='开启较强的红/橙/黄/绿/青/蓝/紫/洋红色相塑形；默认关闭')
    ap.add_argument('--save-json',action='store_true',help='显式保存同名 JSON 参数记录，默认不输出')
    ap.add_argument('--backend',choices=['auto','egl','default'],default='auto')
    ap.add_argument('--allow-software',action='store_true',help='允许 Mesa 软件渲染，仅用于测试')
    args=ap.parse_args()
    if not all(math.isfinite(x) for x in [args.strength,args.exposure,args.sharpness]) or not 0<=args.strength<=2 or not 0<=args.sharpness<=2 or not -3<=args.exposure<=3 or args.max_size<0:
        ap.error('检查参数范围：strength/sharpness 0–2，exposure -3–3，max-size >= 0')
    if min(args.read_workers,args.write_workers,args.queue_size,args.gpu_queue)<1:ap.error('线程数和队列容量必须 >= 1')
    if args.texture_cache_mb<0:ap.error('texture-cache-mb 必须 >= 0')
    root=args.input.expanduser().resolve()
    if root.is_file() and root.suffix.lower() in EXTENSIONS:
        files=[root];output=root.parent
    elif root.is_dir():
        files=sorted(p for p in (root.rglob('*') if args.recursive else root.iterdir()) if p.is_file() and p.suffix.lower() in EXTENSIONS and not p.stem.endswith('_retouched'))
        output=root.with_name(root.name+'_retouched')
    else:ap.error('输入不是支持的图片或文件夹')
    if not files:ap.error('未找到支持的图片')
    try: engine=Retoucher(args.backend,profile=args.profile_gpu,texture_cache_mb=args.texture_cache_mb)
    except Exception as e:ap.exit(1,str(e)+'\n')
    renderer=engine.ctx.info['GL_RENDERER'];print('OpenGL renderer:',renderer)
    if any(s in renderer.lower() for s in ['llvmpipe','softpipe','software','swrast']) and not args.allow_software:
        ap.exit(1,'检测到 CPU 软件渲染；请配置 GPU 驱动。仅测试可加 --allow-software。\n')
    jobs=[]
    for path in files:
        relative=path.relative_to(root) if root.is_dir() else Path(path.name)
        dest=(output/relative).with_name(relative.name+'_retouched.jpg') if root.is_dir() else path.with_name(path.stem+'_retouched.jpg')
        if (dest.exists() or (args.save_json and dest.with_suffix('.json').exists())) and not args.overwrite:
            print('跳过已存在:',dest);continue
        jobs.append((path,dest))
    try:
        return run_pipeline(jobs,engine,args,renderer)
    finally:
        engine.close()


def run_pipeline(jobs,engine,args,renderer):
    """Bounded CPU queues around one context-owning GPU thread (the caller)."""
    from PIL import ImageCms
    srgb=ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
    ready=queue.Queue(maxsize=args.queue_size)
    saving=queue.Queue(maxsize=args.queue_size)
    pending=iter(jobs)
    lock=threading.Lock()
    stop=threading.Event()
    counts={'failed':0,'saved':0}
    diagnostic=getattr(args,'diagnostics',False)
    samples={}
    queue_stats={'input_empty_checks':0,'input_checks':0,'ready_peak':0,'save_peak':0,'inflight_peak':0}
    first_ready=None

    def sample(name,value):
        if diagnostic:
            with lock:samples.setdefault(name,[]).append(value)
    started=time.perf_counter()
    use_process=args.decode_backend=='process' and len(jobs)>1
    decoder_pool=ProcessPoolExecutor(max_workers=args.read_workers,mp_context=mp.get_context('spawn')) if use_process else None

    def release_input(image):
        if isinstance(image,SharedInput):image.close()

    def fail(path,error):
        with lock:
            counts['failed']+=1
            print(f'失败 {path}: {error}',file=sys.stderr,flush=True)

    def put_ready(item):
        blocked=time.perf_counter()
        while not stop.is_set():
            try:
                ready.put(item,timeout=.1)
                sample('reader_enqueue_wait',time.perf_counter()-blocked)
                return
            except queue.Full:pass
        release_input(item[1])

    def reader():
        while not stop.is_set():
            with lock:job=next(pending,None)
            if job is None:return
            path,dest=job
            t=time.perf_counter()
            try:
                if decoder_pool is not None:
                    image=SharedInput(decoder_pool.submit(decode_shared,path,args.max_size).result())
                    prepared={'original_bytes':image.shm.buf} if args.strength else None
                else:
                    image=read_image(path,args.max_size)
                    prepared=prepare_image(image) if args.strength else None
                sample('reader_total',time.perf_counter()-t)
                if isinstance(image,SharedInput):
                    for name,value in image.decode_timings.items():sample(name,value)
                put_ready((job,image,prepared,t,time.perf_counter()-t,None))
            except Exception as error:put_ready((job,None,None,t,0,error))
            # Do not retain a second prepared image while blocked on the next queue item.
            image=None;prepared=None

    def writer():
        while True:
            waiting=time.perf_counter()
            item=saving.get()
            sample('writer_input_wait',time.perf_counter()-waiting)
            try:
                if item is None:return
                result,dest,record=item
                converting=time.perf_counter()
                if isinstance(result,ReadbackPixels):result=result.image()
                sample('rgbx_wrap_check',time.perf_counter()-converting)
                t=time.perf_counter()
                dest.parent.mkdir(parents=True,exist_ok=True)
                result.save(dest,format='JPEG',quality=85,subsampling=2,icc_profile=srgb)
                record['save_seconds']=time.perf_counter()-t
                sample('jpeg_save',record['save_seconds'])
                record['seconds']=time.perf_counter()-record.pop('_started')
                if args.save_json:dest.with_suffix('.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
                with lock:
                    counts['saved']+=1
                    print(f"完成 {record['seconds']:.2f}s（含排队）: {dest}",flush=True)
                    if args.profile_gpu and 'timings' in record['parameters']:
                        print('  同步分段耗时:',record['parameters']['timings'],flush=True)
            except Exception as error:fail(dest,error)
            finally:
                saving.task_done()
                item=None
                result=None

    readers=[threading.Thread(target=reader,name=f'retouch-reader-{i}') for i in range(args.read_workers)]
    writers=[threading.Thread(target=writer,name=f'retouch-writer-{i}') for i in range(args.write_workers)]
    for thread in writers+readers:thread.start()
    print(f'流水线: {args.read_workers} 读取'+('进程（共享内存）' if use_process else '线程')+f' → GPU（在途 {args.gpu_queue} 张）→ {args.write_workers} 保存线程；队列容量 {args.queue_size}',flush=True)
    if diagnostic:
        print(f'[DIAG] CPU={os.cpu_count()} Python={sys.version.split()[0]} OpenGL={engine.ctx.info.get("GL_VERSION")} vendor={engine.ctx.info.get("GL_VENDOR")}',flush=True)
        print(f'[DIAG] images={len(jobs)} max_size={args.max_size} style={args.style} stylize={args.stylize} synchronized_profile={args.profile_gpu}',flush=True)
    inflight=deque()
    waits={'input':0.,'save_queue':0.,'pbo_read':0.}

    def enqueue_save(result,dest,record):
        t=time.perf_counter();saving.put((result,dest,record))
        waits['save_queue']+=time.perf_counter()-t
        queue_stats['save_peak']=max(queue_stats['save_peak'],saving.qsize())

    def collect_one():
        result,dest,record=inflight.popleft()
        t=time.perf_counter()
        try:
            pixels=result.resolve()
            read_seconds=time.perf_counter()-t
            record['parameters']['timings']['pbo_read']=read_seconds
            waits['pbo_read']+=read_seconds
            sample('pbo_collect',read_seconds)
            enqueue_save(pixels,dest,record)
        except Exception as error:fail(record['input'],error)

    try:
        for _ in jobs:
            # If readers are behind, use that gap to drain an outstanding GPU result.
            queue_stats['input_checks']+=1
            if ready.empty():queue_stats['input_empty_checks']+=1
            queue_stats['ready_peak']=max(queue_stats['ready_peak'],ready.qsize())
            if ready.empty() and inflight:collect_one()
            waiting=time.perf_counter()
            job,image,prepared,t,read_seconds,error=ready.get()
            waits['input']+=time.perf_counter()-waiting
            if first_ready is None:first_ready=time.perf_counter()-started
            path,dest=job
            if error is not None:fail(path,error);continue
            try:
                gpu_started=time.perf_counter()
                if args.strength==0:result,params=image.copy(),{}
                else:result,params=engine.process(image,args.style,args.strength,args.exposure,args.sharpness,stylize=args.stylize,prepared=prepared,defer_readback=True)
                gpu_seconds=time.perf_counter()-gpu_started
                sample('gpu_submit',gpu_seconds)
                sample('megapixels',image.size[0]*image.size[1]/1e6)
                for name,value in params.get('timings',{}).items():sample('submit_'+name,value)
                sample('texture_create_upload',params.get('texture_create_upload_seconds',0))
                record=dict(algorithm='shader-retouch-v5-gpu-exposure',input=str(path),output=str(dest),size=list(image.size),style=args.style,stylize=args.stylize,strength=args.strength,sharpness=args.sharpness,exposure_offset=args.exposure,renderer=renderer,parameters=params,read_prepare_seconds=read_seconds,gpu_submit_seconds=gpu_seconds,_started=t)
                prepared=None
                release_input(image);image=None
                if isinstance(result,PendingReadback):
                    inflight.append((result,dest,record))
                    queue_stats['inflight_peak']=max(queue_stats['inflight_peak'],len(inflight))
                    if len(inflight)>=args.gpu_queue:collect_one()
                else:enqueue_save(result,dest,record)
                result=None
            except Exception as error:fail(path,error)
            finally:
                prepared=None
                release_input(image);image=None
        while inflight:collect_one()
    finally:
        for result,_,_ in inflight:result.release()
        stop.set()
        for thread in readers:thread.join()
        while not ready.empty():
            abandoned=ready.get_nowait();release_input(abandoned[1])
        if decoder_pool is not None:decoder_pool.shutdown(wait=True,cancel_futures=True)
        for _ in writers:saving.put(None)
        saving.join()
        for thread in writers:thread.join()
    elapsed=time.perf_counter()-started
    print('主线程累计等待:', ', '.join(f'{k}={v:.3f}s' for k,v in waits.items()),flush=True)
    print(f"批处理完成: 成功 {counts['saved']}，失败 {counts['failed']}，总耗时 {elapsed:.2f}s，吞吐 {counts['saved']/max(elapsed,1e-6):.2f} 张/s",flush=True)
    if diagnostic:
        print('[DIAG] ===== 诊断汇总（请复制此段和上面的总耗时） =====',flush=True)
        print(f'[DIAG] first_input_ready={first_ready or 0:.3f}s（含进程启动） queues={queue_stats}',flush=True)
        for name,value in waits.items():print(f'[DIAG] main_wait_{name}={value:.3f}s wall_fraction={100*value/max(elapsed,1e-6):.1f}%',flush=True)
        for name,values in sorted(samples.items()):
            data=np.asarray(values,dtype=float)
            unit='MP' if name=='megapixels' else 'ms'
            multiplier=1 if unit=='MP' else 1000
            print(f'[DIAG] {name}: n={len(data)} mean={data.mean()*multiplier:.2f}{unit} p50={np.quantile(data,.5)*multiplier:.2f}{unit} p95={np.quantile(data,.95)*multiplier:.2f}{unit} max={data.max()*multiplier:.2f}{unit}',flush=True)
        print(f'[DIAG] texture_pool: hits={engine.texture_hits} misses={engine.texture_misses} evictions={engine.texture_evictions} cached_MiB={engine.texture_pool_bytes/1048576:.1f} budget_MiB={engine.texture_budget/1048576:.1f}',flush=True)
        print('[DIAG] submit_* 是CPU提交耗时，可能含驱动等待；不是GPU执行时间。texture_create_upload 与提交阶段重叠，不能相加。',flush=True)
        print('[DIAG] 多工作线程的累计时间会重叠；主线程等待比例也不等于GPU空闲率。--profile-gpu 会强制同步，不能用于评估正常吞吐。',flush=True)
    return 1 if counts['failed'] else 0

if __name__=='__main__':sys.exit(main())
